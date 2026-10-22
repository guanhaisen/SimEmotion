"""用 savemodel 里已训练好的权重直接跑测试（不重新训练）。

用法（在 SimEmotion 目录、venv 激活下）:
    python test_saved_model.py                  # 自动选 savemodel 里最新的一组权重
    python test_saved_model.py 20261010093154   # 指定时间戳（见 savemodel 文件名）
    python test_saved_model.py 20261021130957 --datadir FI_seed43   # 指定数据划分目录

说明:
- 复用 main.py 的数据/模型定义（Loaddata / FCN / get_features_test），
  保证与训练时的测试逻辑完全一致；
- --datadir 必须与训练时一致（如 seed43 实验用 FI_seed43），否则跨划分测出的
  精度因训练集/测试集图片重叠而虚高，不可与训练日志对照；
- 注意 savemodel 里保存的是 "best-val checkpoint"，不是最后一轮模型；
- 跑完会打印混淆矩阵和 Test Acc，与训练日志中的数字应当一致。
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


def find_checkpoint(ts=None):
    if ts is None:
        cands = sorted(glob.glob(os.path.join(SAVE_DIR, "modelViT-B-16_FI_8_*.pth")))
        if not cands:
            raise SystemExit("savemodel 里没有找到模型文件")
        mpath = cands[-1]  # 文件名带时间戳，排序后最后一个就是最新的
        ts = os.path.basename(mpath)[:-4].split("_")[-1]
    else:
        mpath = os.path.join(SAVE_DIR, f"modelViT-B-16_FI_8_{ts}.pth")
    fpath = os.path.join(SAVE_DIR, f"fcViT-B-16_FI_8_{ts}.pth")
    if not (os.path.exists(mpath) and os.path.exists(fpath)):
        raise SystemExit(f"找不到权重: {mpath} / {fpath}")
    return mpath, fpath


def main():
    parser = argparse.ArgumentParser(description='用已保存的权重跑测试集（不重新训练）')
    parser.add_argument('timestamp', nargs='?', default=None, help='checkpoint 时间戳（savemodel 文件名末段），缺省取最新')
    parser.add_argument('--datadir', type=str, default=None, help='数据划分目录（如 FI_seed43），须与训练时一致；缺省用 FI/')
    args = parser.parse_args()

    mpath, fpath = find_checkpoint(args.timestamp)
    print("model 权重:", mpath)
    print("fc 权重   :", fpath)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ---- 与 main.py 完全相同的初始化顺序 ----
    M.set_seed(1)
    # args 必须先于 datasetInit 设置（datasetInit 内部读取 args.datadir）
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'   # 与 main.py 相同：路径按字符串拼接，结尾斜杠必须补齐
    M.args = types.SimpleNamespace(datasetname="FI_8", batch_size=32, datadir=args.datadir)
    M.classtype, M.data_dir = M.datasetInit("FI_8")
    M.device = device
    M.criterion = torch.nn.CrossEntropyLoss()
    M.log = []
    print("data_dir:", M.data_dir)

    model, preprocess = clip.load("ViT-B/16", device)
    clip.model.convert_weights(model)
    model.load_state_dict(torch.load(mpath, map_location="cpu"))
    model = model.to(device).eval()
    M.preprocess = preprocess

    fc = M.FCN(dim=512).to(device)
    fc.load_state_dict(torch.load(fpath, map_location="cpu"))
    fc.eval()
    M.model, M.fc = model, fc

    # ---- 跑测试集 ----
    testset = M.Loaddata("test")
    print("test 集图片数:", len(testset))
    acc = M.get_features_test(testset)
    print("FINAL TEST ACC =", round(100 * acc, 4))


if __name__ == "__main__":
    main()