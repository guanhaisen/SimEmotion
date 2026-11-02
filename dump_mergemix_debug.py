"""dump_mergemix_debug.py — MergeMix 增强可视化（CPU 即可运行，不需要 GPU）

取 1 个真实训练 batch，走 mergemix 的完整增强流程（ToMe 补丁 → 注意力恢复 →
TopK 掩码 → 像素混合），把 原图i / 原图j(伙伴) / 掩码 / 混合图 拼成网格存到
./log/mergemix_debug_<ts>.png，并在终端打印 λ、λ̂ 的数值。

用法（SimEmotion 目录、venv 激活下）:
  python dump_mergemix_debug.py
  python dump_mergemix_debug.py --mm_r 8 --n_show 4
"""
import argparse
import types

import clip
import torch
from torchvision import transforms
from PIL import Image

import main as M
import tome_clip


def main():
    parser = argparse.ArgumentParser(description='MergeMix augmentation visualizer')
    parser.add_argument('--mm_r', type=int, default=8, help='ToMe 每层合并 token 数')
    parser.add_argument('--mm_alpha', type=float, default=1.0, help='Beta(α,α) 参数')
    parser.add_argument('--n_show', type=int, default=4, help='展示前 n 个样本')
    parser.add_argument('--datadir', type=str, default='FI_seed43', help='数据划分目录')
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'

    device = 'cpu'
    M.args = types.SimpleNamespace(datasetname='FI_8', batch_size=32, datadir=args.datadir)
    classtype, data_dir = M.datasetInit('FI_8')
    M.classtype = classtype
    M.data_dir = data_dir
    M.device = device
    M.set_seed(args.seed)

    model, preprocess = clip.load('ViT-B/16', device)
    tome_clip.apply_patch(model.visual, r=args.mm_r, trace_source=True)
    model.eval()

    trainset = M.Loaddata('train')
    images, labels = next(iter(torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True)))
    B = args.n_show
    images = images[:B]

    # ---- 与 supcon.py train_joint mergemix 分支完全一致的增强流程 ----
    with torch.no_grad():
        model.visual.r = args.mm_r
        model.encode_image(images)                       # 干净图前向（快照注意力）
        info = model.visual._tome_info
        attn_L = tome_clip.recover_cls_attention(info)   # (B, 197) Eq.6
        k_final = info["source"].shape[-1] if info["source"] is not None else 197
        lam = torch.distributions.Beta(args.mm_alpha, args.mm_alpha).sample((B,))
        mixed, perm, mask_tok = tome_clip.build_mix(images, attn_L, lam)   # Eq.2/7
        lam_hat = tome_clip.rescale_lambda(k_final, attn_L.shape[1], mask_tok)  # Eq.8

    # ---- 打印数值 ----
    names = [classtype[l] for l in labels[:B]]
    print(f"mm_r={args.mm_r}  K_final={k_final}  (197 token, 压缩 {1 - k_final / 197:.0%})")
    for b in range(B):
        print(f"样本{b}: {names[b]:<12} λ={lam[b]:.3f}  λ̂={lam_hat[b]:.3f}  "
              f"掩码 {int(mask_tok[b].sum())}/196 patch  伙伴=样本{int(perm[b])}({classtype[labels[int(perm[b])]]})")

    # ---- 拼图: 每行 = [原图i | 掩码叠加 | 混合图 | 伙伴原图j] ----
    inv_norm = transforms.Normalize(
        mean=[-(0.48145466 / 0.26862954), -(0.4578275 / 0.26130258), -(0.40821073 / 0.27577711)],
        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711])
    to_img = lambda t: (inv_norm(t).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

    from PIL import Image as PImage
    import numpy as np
    rows = []
    for b in range(B):
        base = to_img(images[b])
        mask_px = (mask_tok[b].view(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1))
        overlay = base.copy()
        overlay[mask_px > 0.5] = [255, 80, 80]           # 掩码区域标红
        j = int(perm[b])
        rows.append(np.concatenate([base, overlay, to_img(mixed[b]), to_img(images[j])], axis=1))
    grid = np.concatenate(rows, axis=0)
    import os, time
    os.makedirs('./log', exist_ok=True)
    ts = time.strftime('%Y%m%d%H%M%S')
    out = f'./log/mergemix_debug_{ts}.png'
    PImage.fromarray(grid).save(out)
    print(f"\n已保存: {out}")
    print("每行 4 列 = [原图 i | 掩码叠加(红=取自 i) | 混合图 x̂ | 伙伴原图 j]")
    print("读法: 混合图中红色区域来自原图 i，其余来自伙伴 j；λ 越大红区越多")


if __name__ == '__main__':
    main()
