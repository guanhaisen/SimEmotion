"""wise_ft_merge.py — WiSE-FT 权重插值 α 扫描（论文本体：weight-space ensembling，零训练成本）

用法（SimEmotion 目录、venv 激活下）：
  # wiseft-full 臂（20261102153748，默认 α ∈ {0, 0.3, 0.5, 0.7, 0.9, 1.0}）
  python wise_ft_merge.py 20261102153748 --datadir FI_seed43 --wise_init

  # 自定义扫描列表
  python wise_ft_merge.py 20261102153748 --datadir FI_seed43 --wise_init --alphas 0,0.3,0.5,0.7,0.9,1.0

  # 缺省时间戳自动取 savemodel 最新两件套
  python wise_ft_merge.py --datadir FI_seed43 --wise_init

原理（Wortsman et al. 2022, Eq.1）：
  θ(α) = (1−α)·θ₀ + α·θ₁   逐键线性插值
  θ₀ = OpenAI CLIP 预训练权重（visual encoder）+ 零样本文本嵌入分类器（main.build_zero_shot_head，
       与 --wise_init 完全同一条 ZS_TEMPLATES 管线）
  θ₁ = main.py 微调 checkpoint（model + fc 两件套）
  α=0 → 纯零样本；α=1 → 纯微调（应复现该臂 FinalTest，作为管线自检）；α∈(0,1) → 权重空间 ensemble

要点：
- 纯内存合并，不产生中间 checkpoint 文件；每个 α 一次测试集前向（~3500 张，建议 GPU）
- 评估协议必须与训练臂一致：--wise_init 训练的臂必须加 --wise_init（评估特征 L2 归一化），
  两个端点都定义在归一化特征上，插值模型沿同一条协议评估
- fc 插值要求键一致：wise_init 臂的 fc 是单层线性头（classifier.weight/bias），与零样本分类器
  同构可直接插值；MLP 臂的 fc 键不匹配时自动退化为"仅编码器插值"（fc 固定用微调权重）并打印警告
- 结果打印汇总表并写入 result/log_wisefull_merge_<ts>.txt（含每个 α 的混淆矩阵与 checkpoint 行）
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
RESULT_DIR = os.path.join(SCRIPT_DIR, "result")


def find_checkpoints(ts, datasetname):
    """main.py 系两件套：modelViT-B-16_<datasetname>_<ts>.pth / fc 同构；ts 缺省取最新"""
    if ts is None:
        cands = sorted(glob.glob(os.path.join(SAVE_DIR, f"modelViT-B-16_{datasetname}_*.pth")))
        if not cands:
            raise SystemExit(f"savemodel 里没有找到 {datasetname} 的模型文件")
        ts = os.path.basename(cands[-1])[:-4].split("_")[-1]
    mpath = os.path.join(SAVE_DIR, f"modelViT-B-16_{datasetname}_{ts}.pth")
    fpath = os.path.join(SAVE_DIR, f"fcViT-B-16_{datasetname}_{ts}.pth")
    if not (os.path.exists(mpath) and os.path.exists(fpath)):
        raise SystemExit(f"找不到权重: {mpath} / {fpath}")
    return mpath, fpath


def merge_state_dicts(sd0, sd1, alpha):
    """θ(α) = (1−α)·θ₀ + α·θ₁，逐键 fp32 计算后还原原 dtype；α=0/1 精确回到对应端点
    sd0 可能在 GPU（从显存模型克隆）、sd1 在 CPU（磁盘加载）→ 统一搬到 t0 的设备再算"""
    if set(sd0.keys()) != set(sd1.keys()):
        raise SystemExit("两套 state_dict 键不一致，无法插值")
    merged = {}
    for k, t0 in sd0.items():
        t1 = sd1[k].to(t0.device)
        merged[k] = ((1.0 - alpha) * t0.float() + alpha * t1.float()).to(t0.dtype)
    return merged


def main():
    parser = argparse.ArgumentParser(description='WiSE-FT 权重插值 α 扫描（零训练成本 ensemble）')
    parser.add_argument('timestamp', nargs='?', default=None,
                        help='checkpoint 时间戳（savemodel 文件名末段），缺省取最新')
    parser.add_argument('--datadir', type=str, default=None,
                        help='数据划分目录（如 FI_seed43），须与训练时一致')
    parser.add_argument('--datasetname', type=str, default='FI_8',
                        help='数据集名（FI_8 / ArtPhoto / EmotionROI）')
    parser.add_argument('--alphas', type=str, default='0,0.3,0.5,0.7,0.9,1.0',
                        help='逗号分隔的混合系数扫描列表；0=纯零样本，1=纯微调自检点')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wise_init', action='store_true',
                        help='微调臂为 main.py --wise_init 训练时必须加（评估特征 L2 归一化，与训练协议一致）')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    alphas = [float(a) for a in args.alphas.split(',') if a.strip() != '']
    if not alphas or any(a < 0 or a > 1 for a in alphas):
        raise SystemExit("alpha 列表为空或含 [0,1] 以外的值")

    # ---- 与 main.py / test_saved_model.py 相同的初始化顺序 ----
    M.set_seed(1)
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'   # 路径按字符串拼接，结尾斜杠必须补齐
    M.args = types.SimpleNamespace(datasetname=args.datasetname, batch_size=args.batch_size,
                                   datadir=args.datadir, wise_init=args.wise_init)
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

    # ---- 两套端点权重 ----
    mpath, fpath = find_checkpoints(args.timestamp, args.datasetname)
    print("微调权重（θ₁）:", mpath)
    sd1_model = torch.load(mpath, map_location="cpu")
    sd1_fc = torch.load(fpath, map_location="cpu")
    # θ₀：OpenAI CLIP 预训练权重（fp16 原样克隆）+ 零样本文本分类器（与 --wise_init 同管线）
    sd0_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    fc_zs = M.build_zero_shot_head(model, device, 512)
    sd0_fc = fc_zs.state_dict()
    # fc 键匹配检查：wise_init 臂（单层线性头）可直接插值；MLP 臂退化为仅编码器插值
    fc_mergeable = set(sd0_fc.keys()) == set(sd1_fc.keys())
    if not fc_mergeable:
        print("警告: fc 键不匹配（MLP 头 vs 零样本线性分类器）→ 仅插值编码器，fc 固定用微调权重")

    testset = M.Loaddata('test')
    print("test 集图片数:", len(testset))

    model = model.to(device)
    results = []
    for alpha in alphas:
        model.load_state_dict(merge_state_dicts(sd0_model, sd1_model, alpha))
        if fc_mergeable:
            fc_sd = merge_state_dicts(sd0_fc, sd1_fc, alpha)
        else:
            fc_sd = sd1_fc
        fc = M.FCN(dim=512, linear_head=('classifier.weight' in fc_sd)).to(device)
        fc.load_state_dict(fc_sd)
        fc.eval()
        M.model, M.fc = model.eval(), fc
        head = f'WSE(a={alpha:g})'
        print(f'== {head} ==')
        acc = M.get_features_test(testset, prefix=head)
        results.append((alpha, acc))

    # ---- 汇总表 ----
    print('\n== α 扫描汇总 ==')
    print('α       Test Acc')
    best_a, best_acc = None, -1.0
    for alpha, acc in results:
        print(f'{alpha:<7g} {round(100 * acc, 4)}')
        if acc > best_acc:
            best_a, best_acc = alpha, acc
    print(f'best: α={best_a:g} → {round(100 * best_acc, 4)}%')
    print('自检: α=1 应等于该臂 FinalTest；α=0 为零样本基线（情绪任务预期显著偏低）')

    # ---- 日志归档 ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    logpath = os.path.join(RESULT_DIR, f'log_wisefull_merge_{args.timestamp or "latest"}.txt')
    with open(logpath, 'a+', encoding='utf-8') as f:
        f.write('wise_ft_merge α 扫描  dataset:{} datadir:{} wise_init:{} batch:{} alphas:{}\n'.format(
            args.datasetname, args.datadir, args.wise_init, args.batch_size, args.alphas))
        f.write('checkpoint: ' + os.path.basename(mpath) + ' / ' + os.path.basename(fpath) + '\n')
        for line in M.log:
            f.write(str(line))
        f.write('summary: ' + '; '.join(f'a={a:g}:{round(100 * acc, 4)}' for a, acc in results)
                + f'; best a={best_a:g}\n')
    print('日志已写入:', logpath)


if __name__ == "__main__":
    main()
