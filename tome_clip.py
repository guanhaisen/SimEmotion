"""tome_clip.py — OpenAI CLIP 视觉编码器的 ToMe（Token Merging）移植补丁 + MergeMix 掩码/混合工具

为什么自己移植：facebookresearch/ToMe 官方库只提供 timm/swag/mae 补丁，没有 OpenAI CLIP 补丁；
MergeMix（Jin et al., ICLR 2026）原文也是自己把注意力层替换为 ToMeAttention。
本文件把官方 tome/merge.py 的核心算法（BSD 风格 Meta 许可）逐行移植，并适配 OpenAI CLIP 的
ResidualAttentionBlock（LND 张量格式 + nn.MultiheadAttention 打包 QKV 权重）。

对官方 ToMe 的实现差异（写实验记录时必须注明）:
  - metric（相似度度量）：官方 timm 版返回多头 k 的头均值；本移植用同样的"打包投影 k 按头均值"，
    数学上与官方一致（k = x @ W_k^T 后按 12 头 reshape 取均值）
  - proportional attention（size.log() 加进 attn logits）与 merge_wavg（按 token 尺寸加权平均）
    均按官方逐行移植，r>0 时生效，r=0 时数学上退化为原始 CLIP 前向
  - 注意力重实现（F.linear 拆 QKV + 手工多头注意力）与 nn.MultiheadAttention 数值等价，
    已通过 fp32 CPU 逐元素对照测试（rtol 1e-5）

MergeMix 专用工具（对应论文 Eq.5-9）:
  - recover_cls_attention(info): A_L = R_{K→L}(A_K, S)，用 source map 把合并分辨率注意力恢复到原始序列
  - build_mix(): TopK 掩码（Eq.7, p=⌊λ·L⌋）→ 像素级 patch 混合（Eq.2）
  - rescale_lambda(): 高斯重缩放混合比例（Eq.8, μ=K/L, σ=ΣM/L, τ=1e-5）
"""
import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


# ==================== 核心算法（移植自 facebookresearch/ToMe tome/merge.py） ====================

def do_nothing(x, mode=None):
    return x


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = True,
) -> Tuple[Callable, Callable]:
    """ToMe 二分软匹配（balanced 50/50 划分）。
    metric: [batch, tokens, channels]；r 为本层要移除的 token 数（上限 (t-protected)//2）。
    class_token=True 时保护首位 CLS token 不参与合并。返回 (merge, unmerge) 闭包。"""
    protected = 1 if class_token else 0
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)
    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)
        if class_token:
            scores[..., 0, :] = -math.inf
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]   # 未合并 token
        src_idx = edge_idx[..., :r, :]   # 被合并 token
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]  # 保证 CLS 排在最前

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))
        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)
        return out

    return merge, unmerge


def merge_wavg(merge: Callable, x: torch.Tensor, size: torch.Tensor = None):
    """按 token 尺寸加权平均合并（官方同名函数移植）。返回 (合并后 x, 新 token 尺寸)。"""
    if size is None:
        size = torch.ones_like(x[..., 0, None])
    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def merge_source(merge: Callable, x: torch.Tensor, source: torch.Tensor = None) -> torch.Tensor:
    """source map 追踪（官方同名函数移植）：初始 token 与最终合并组的邻接矩阵 (N, L, K)。"""
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)
    source = merge(source, mode="amax")
    return source


def tokens_after_merging(L: int, r: int, n_layers: int) -> int:
    """给定每层合并数 r，计算最终 token 数 K（用于 λ̂ 重缩放的 μ=K/L）。"""
    t = L
    for _ in range(n_layers):
        rr = min(r, (t - 1) // 2)
        if rr <= 0:
            break
        t -= rr
    return t


# ==================== OpenAI CLIP 补丁 ====================

def _make_tome_block(block_cls):
    """生成 ToMe 版 ResidualAttentionBlock 类（完整版，metric 在 forward 内产出）。"""

    class ToMeResidualAttentionBlock(block_cls):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            info = self._tome_info
            x_in = self.ln_1(x)
            qkv = F.linear(x_in, self.attn.in_proj_weight, self.attn.in_proj_bias)  # (L,N,3D)
            D = self.attn.embed_dim
            H = self.attn.num_heads
            dh = D // H
            scale = dh ** -0.5
            L, N, _ = qkv.shape
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(L, N, H, dh).permute(1, 2, 0, 3)   # (N,H,L,dh)
            k = k.reshape(L, N, H, dh).permute(1, 2, 0, 3)
            v = v.reshape(L, N, H, dh).permute(1, 2, 0, 3)
            attn = (q @ k.transpose(-2, -1)) * scale
            size = info["size"]
            if size is not None:
                attn = attn + size.log()[:, None, None, :, 0]
            attn = attn.softmax(dim=-1)
            info["attn_cls"] = attn[:, :, 0, :].mean(1)       # (N, L_cur) 最后一层覆盖
            info["source_attn"] = info["source"]              # 与 attn 同分辨率的 source 快照
            # metric：多头 k 的头均值（官方 ToMeAttention 同口径），供本层合并使用
            info["metric"] = k.mean(1)                        # (N, L_cur, dh)
            attn_out = (attn @ v).permute(2, 0, 1, 3).reshape(L, N, D)
            attn_out = self.attn.out_proj(attn_out)
            x = x + attn_out
            r = info["r"].pop(0)
            if r > 0:
                xt = x.permute(1, 0, 2)                       # LND -> NLD
                merge, _ = bipartite_soft_matching(info["metric"], r, class_token=True)
                if info["trace_source"]:
                    info["source"] = merge_source(merge, xt, info["source"])
                xt, info["size"] = merge_wavg(merge, xt, info["size"])
                x = xt.permute(1, 0, 2)                       # NLD -> LND
            x = x + self.mlp(self.ln_2(x))
            return x

    return ToMeResidualAttentionBlock


def apply_patch(visual, r: int = 0, trace_source: bool = True):
    """给 OpenAI CLIP 的 visual 编码器打 ToMe 补丁（原地改 __class__，不复制任何权重）。

    用法:
        import tome_clip
        tome_clip.apply_patch(model.visual, r=8)
        model.visual.r = 8    # 训练时每层合并数；评估时置 0（数学上完全退化为原始前向）
        features = model.encode_image(images)   # 前向结束后 model.visual._tome_info 里有:
        info["attn_cls"]     # 最后一层 CLS→token 注意力 (N, K_attn)，K_attn 为最后一次合并前的 token 数
        info["source_attn"]  # 与上者同分辨率的 source map (N, K_attn, L) 或 None
        info["source"]       # 前向结束时的最终 source map (N, K_final, L)
    """
    import clip.model as cm

    visual_cls = type(visual)

    class ToMeVisionTransformer(visual_cls):
        """每次 forward 前重置 r 调度 / size / source（官方 ToMeVisionTransformer 同逻辑）。"""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n_layers = len(self.transformer.resblocks)
            self._tome_info["r"] = [getattr(self, "r", 0)] * n_layers
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            self._tome_info["attn_cls"] = None
            self._tome_info["source_attn"] = None
            return super().forward(x)

    visual.__class__ = ToMeVisionTransformer
    visual.r = r
    visual._tome_info = {
        "r": [r],
        "size": None,
        "source": None,
        "attn_cls": None,
        "source_attn": None,
        "trace_source": trace_source,
        "metric": None,
    }
    tome_block_cls = _make_tome_block(cm.ResidualAttentionBlock)
    for module in visual.transformer.resblocks:
        module.__class__ = tome_block_cls
        module._tome_info = visual._tome_info
    return visual


# ==================== MergeMix 工具（论文 Eq.5-9） ====================

@torch.no_grad()
def recover_cls_attention(info: dict) -> torch.Tensor:
    """A_L = R_{K→L}(A_K, S)（论文 Eq.6）：把合并分辨率的 CLS 注意力恢复到原始 token 序列。
    ToMe source map 约定（官方 merge_source）：(N, K_cur, L_initial)，source[n,k,l]=1 表示
    初始 token l 属于当前组 k。故 A_L = A_K @ S。返回 (N, L=1+grid²)（第 0 位是 CLS）。
    r=0 全程时 source_attn 为 None，注意力本来就在原始分辨率，直接返回。"""
    attn = info["attn_cls"]          # (N, K)
    src = info["source_attn"]        # (N, K, L) or None
    if src is None:
        return attn
    return torch.bmm(attn.unsqueeze(1).to(src.dtype), src).squeeze(1)     # (N, 1, K)@(N,K,L) -> (N, L)


@torch.no_grad()
def build_mix(images: torch.Tensor, attn_L: torch.Tensor, lam: torch.Tensor,
              grid: int = 14, img_size: int = 224, mask_mode: str = 'attn'):
    """TopK 掩码 + patch 级像素混合（论文 Eq.2/Eq.7）。
    images: (B,3,H,W)；attn_L: (B, 1+grid²) 恢复后的 CLS 注意力（第 0 位是 CLS）；
    lam: (B,) 每样本混合比例。返回 (mixed, perm, mask_tok)。
    perm 为混合伙伴置换，mixed[b] = M[b]⊙x[b] + (1-M[b])⊙x[perm[b]]。
    mask_mode:
      'attn' —— 注意力 TopK 选区（MergeMix 论文原版，Eq.7）
      'rect' —— CutMix 式随机矩形选区（消融对照：单变量检验"注意力选区 vs 随机选区"，
                面积同为 ⌊λ·196⌋，其余管线不动；矩形内取自 x_i、外部取自伙伴 x_j，与 CutMix 原版同向）"""
    B = images.size(0)
    n_patch = grid * grid
    if mask_mode == 'rect':
        ps = torch.floor(lam.to(images.device) * n_patch).long().clamp(0, n_patch)
        mask_tok = torch.zeros(B, n_patch, device=images.device)
        for b in range(B):
            side = int(round(float(ps[b]) ** 0.5))
            side = max(0, min(side, grid))
            if side > 0:
                r0 = int(torch.randint(0, grid - side + 1, (1,)).item())
                c0 = int(torch.randint(0, grid - side + 1, (1,)).item())
                mask_tok[b].view(grid, grid)[r0:r0 + side, c0:c0 + side] = 1.0
    else:
        patch_attn = attn_L[:, 1:].float()                    # (B, 196)
        ps = torch.floor(lam.to(patch_attn.device) * n_patch).long().clamp(0, n_patch)
        mask_tok = torch.zeros_like(patch_attn)
        for b in range(B):
            if ps[b] > 0:
                idx = patch_attn[b].topk(int(ps[b])).indices
                mask_tok[b, idx] = 1.0
    # token 掩码 -> 像素掩码（每 patch 覆盖 16×16 px）
    m = mask_tok.view(B, 1, grid, grid)
    m = m.repeat_interleave(img_size // grid, dim=2).repeat_interleave(img_size // grid, dim=3)
    perm = torch.randperm(B, device=images.device)
    mixed = m * images + (1.0 - m) * images[perm]
    return mixed, perm, mask_tok


def rescale_lambda(k_final: int, L: int, mask_tok: torch.Tensor,
                   tau: float = 1e-5) -> torch.Tensor:
    """混合比例高斯重缩放（论文 Eq.8）：λ̂~N(μ=K/L, σ=ΣM_i/L)，
    再 batch 内 min-max 归一化（分母 + τ）并 clip 到 [0,1]。返回 (B,) 的 λ̂。"""
    B = mask_tok.size(0)
    mu = torch.full((B,), float(k_final) / float(L), device=mask_tok.device)
    sigma = mask_tok.sum(dim=1) / float(L)
    lam_raw = torch.randn(B, device=mask_tok.device) * sigma + mu
    lam_hat = (lam_raw - lam_raw.min()) / (lam_raw.max() - lam_raw.min() + tau)
    return lam_hat.clamp(0.0, 1.0)
