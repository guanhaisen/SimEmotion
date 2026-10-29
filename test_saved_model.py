"""test_saved_model.py — 用已训练好的权重直接跑测试（不重新训练）。

用法（SimEmotion 目录、venv 激活下）:
  # main.py 系（无 tag）：
  python test_saved_model.py                                  # 自动选最新的一组权重（FI_8）
  python test_saved_model.py 20261010093154                   # 指定 main.py 时间戳
  python test_saved_model.py 20261021130957 --datadir FI_seed43
  python test_saved_model.py 20261023202902 --datasetname ArtPhoto --datadir artphoto_split

  # supcon.py 系（--tag 指定标记，checkpoint 命名为 model<tag>_<dataset>_<ts>.pth，无 ViT-B-16 段）：
  python test_saved_model.py 20261026152557 --tag supconpaco  --datadir FI_seed43          # 三件套走 fc 路径
  python test_saved_model.py 20261027152851 --tag supconspread --datadir FI_seed43
  python test_saved_model.py 20261029131417 --tag supconpure  --datadir emotionroi_split   # 两件套自动切 kNN

  # 跨数据集（如 FI 模型测 ArtPhoto——图像源不同，无泄漏问题）：
  python test_saved_model.py 20261016132806 --datadir artphoto_split

说明:
- 复用 main.py 的数据/模型定义（Loaddata / FCN / get_features_test / f1_from_matrix），
  保证与训练时的测试逻辑完全一致；
- --datadir 必须与训练时一致（如 seed43 实验用 FI_seed43），否则跨划分测出的
  精度因训练集/测试集图片重叠而虚高，不可与训练日志对照；
- 注意 savemodel 里保存的是 "best-val checkpoint"，不是最后一轮模型；
- 跑完会打印混淆矩阵和 Test Acc，与训练日志中的数字应当一致；
- --tag 为 supcon 系标记（supconjoint/supcon2stage/supconpure/supcongenscl/supconpaco/
  supconspread/supconpacoplus）：三件套加载 model+fc 走 fc 评估路径（proj 不参与评估）；
  两件套（supconpure，无 fc）加载 model+proj 自动切换 kNN 评估路径（复用 supcon.py 的
  extract_z / knn_classify，余弦相似度加权投票）。
"""
import argparse
import glob
import os
import types

import clip
import torch

import main as M

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "savemodel")


def find_main_checkpoint(ts, datasetname):
    """main.py 系：modelViT-B-16_<datasetname>_<ts>.pth / fc 同构；ts 缺省取最新"""
    if ts is None:
        cands = sorted(glob.glob(os.path.join(SAVE_DIR, f"modelViT-B-16_{datasetname}_*.pth")))
        if not cands:
            raise SystemExit(f"savemodel 里没有找到 {datasetname} 的模型文件")
        mpath = cands[-1]  # 文件名带时间戳，排序后最后一个就是最新的
        ts = os.path.basename(mpath)[:-4].split("_")[-1]
    mpath = os.path.join(SAVE_DIR, f"modelViT-B-16_{datasetname}_{ts}.pth")
    fpath = os.path.join(SAVE_DIR, f"fcViT-B-16_{datasetname}_{ts}.pth")
    if not (os.path.exists(mpath) and os.path.exists(fpath)):
        raise SystemExit(f"找不到权重: {mpath} / {fpath}")
    return mpath, fpath


def find_supcon_checkpoint(ts, datasetname, tag):
    """supcon.py 系：model<tag>_<datasetname>_<ts>.pth（注意无 ViT-B-16 段，save_checkpoints 用 model{tag} 拼接）
    返回 (model, fc或None, proj或None)"""
    if ts is None:
        cands = sorted(glob.glob(os.path.join(SAVE_DIR, f"model{tag}_{datasetname}_*.pth")))
        if not cands:
            raise SystemExit(f"savemodel 里没有找到 tag={tag} 的模型文件")
        mpath = cands[-1]
        ts = os.path.basename(mpath)[:-4].split("_")[-1]
    mpath = os.path.join(SAVE_DIR, f"model{tag}_{datasetname}_{ts}.pth")
    fpath = os.path.join(SAVE_DIR, f"fc{tag}_{datasetname}_{ts}.pth")
    ppath = os.path.join(SAVE_DIR, f"proj{tag}_{datasetname}_{ts}.pth")
    if not os.path.exists(mpath):
        raise SystemExit(f"找不到权重: {mpath}")
    has_fc = os.path.exists(fpath)
    has_proj = os.path.exists(ppath)
    if not has_fc and not has_proj:
        raise SystemExit(f"找不到 fc 或 proj 权重（{fpath} / {ppath} 均不存在）")
    return mpath, fpath if has_fc else None, ppath if has_proj else None


def run_pure_knn_eval(args, classtype, model, proj, device):
    """pure 两件套（无 fc）的 kNN 评估路径：复用 supcon.py 的 extract_z/knn_classify/knn_evaluate"""
    import supcon as S   # 局部导入：仅在需要 kNN 路径时引入

    trainset = M.Loaddata("train")
    testset = M.Loaddata("test")
    log = []
    print("kNN 评估（余弦相似度加权投票，k={}）：先编码训练集，再对测试集投票……".format(args.knn_k))
    S.knn_evaluate(model, proj, trainset, testset, classtype,
                   args.batch_size, device, args.knn_k, log, prefix='FinalTest')


def main():
    parser = argparse.ArgumentParser(description='用已保存的权重跑测试集（不重新训练）')
    parser.add_argument('timestamp', nargs='?', default=None,
                        help='checkpoint 时间戳（savemodel 文件名末段），缺省取最新')
    parser.add_argument('--datadir', type=str, default=None,
                        help='数据划分目录（如 FI_seed43 / artphoto_split），须与训练时一致；缺省用 FI/')
    parser.add_argument('--datasetname', type=str, default='FI_8',
                        help='数据集名（FI_8 / ArtPhoto / EmotionROI），须与训练时一致')
    parser.add_argument('--tag', type=str, default=None,
                        help='supcon.py 系标记（supconjoint/supconpure/supcongenscl/supconpaco/'
                             'supconspread/supconpacoplus/supcon2stage）；缺省走 main.py 系权重')
    parser.add_argument('--knn_k', type=int, default=20, help='pure 两件套的 kNN 邻居数')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- 与 main.py 完全相同的初始化顺序 ----
    M.set_seed(1)
    # args 必须先于 datasetInit 设置（datasetInit 内部读取 args.datadir）
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'   # 与 main.py 相同：路径按字符串拼接，结尾斜杠必须补齐
    M.args = types.SimpleNamespace(datasetname=args.datasetname, batch_size=32, datadir=args.datadir)
    classtype, data_dir = M.datasetInit(args.datasetname)
    M.classtype = classtype
    M.data_dir = data_dir
    M.device = device
    M.criterion = torch.nn.CrossEntropyLoss()
    M.log = []
    print("data_dir:", M.data_dir)

    model, preprocess = clip.load("ViT-B/16", device)
    clip.model.convert_weights(model)
    M.preprocess = preprocess

    if args.tag is None:
        # ---- main.py 系：model + fc，标准 fc 评估路径 ----
        mpath, fpath = find_main_checkpoint(args.timestamp, args.datasetname)
        print("model 权重:", mpath)
        print("fc 权重   :", fpath)
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model = model.to(device).eval()
        fc = M.FCN(dim=512).to(device)
        fc.load_state_dict(torch.load(fpath, map_location="cpu"))
        fc.eval()
        M.model, M.fc = model, fc
        testset = M.Loaddata("test")
        print("test 集图片数:", len(testset))
        acc = M.get_features_test(testset)
        print("FINAL TEST ACC =", round(100 * acc, 4))
    else:
        # ---- supcon.py 系：三件套走 fc 路径；两件套（pure）自动切 kNN ----
        mpath, fpath, ppath = find_supcon_checkpoint(args.timestamp, args.datasetname, args.tag)
        print("model 权重:", mpath)
        model.load_state_dict(torch.load(mpath, map_location="cpu"))
        model = model.to(device).eval()
        M.model = model
        if fpath is not None:
            print("fc 权重   :", fpath, "（三件套，标准 fc 评估路径；proj 不参与评估）")
            fc = M.FCN(dim=512).to(device)
            fc.load_state_dict(torch.load(fpath, map_location="cpu"))
            fc.eval()
            M.fc = fc
            testset = M.Loaddata("test")
            print("test 集图片数:", len(testset))
            acc = M.get_features_test(testset)
            print("FINAL TEST ACC =", round(100 * acc, 4))
        else:
            print("proj 权重 :", ppath, "（两件套无 fc，切换 kNN 评估路径）")
            proj = S_Proj().to(device)
            proj.load_state_dict(torch.load(ppath, map_location="cpu"))
            proj.eval()
            run_pure_knn_eval(args, classtype, model, proj, device)


def S_Proj():
    """延迟导入 supcon 的投影头（仅 pure 两件套 kNN 路径需要）"""
    import supcon as S
    return S.SupConProjector(dim=512)


if __name__ == "__main__":
    main()
