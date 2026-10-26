"""supcon.py — 监督对比损失（SupCon, Khosla et al., NeurIPS 2020）实验脚本

四种范式（--mode 选择）:
  joint    (默认) : L = CE(fc(features)) + w·SupCon(proj(features))，单循环，
                    与 main.py 七档 loss 臂完全同构（同 warmup/双优化器/调度/best-val），消融对照最公平
  twostage        : 文献标准两阶段——Stage1 SupCon 预训练 backbone+投影头，Stage2 冻结特征用 CE 训分类头
  pure            : 纯 SupCon——全程无任何 CE 梯度，分类用 kNN（余弦相似度加权投票）评估，
                    最纯粹地度量对比损失对特征空间的塑造质量（文献标准的非参数评估）
  genscl          : 广义监督对比损失（GenSCL, Kim et al. arXiv:2206.00384 Eq.2）忠实复刻，
                    训练骨架与 joint 完全一致，仅把 SupCon 项换成"标签相似度 × 特征相似度交叉熵"
                    （one-hot 标签下与 SupCon 的唯一数学差异：逐 anchor 归一化 1/|A(i)| vs 1/|P(i)|）
  paco            : 官方口径 PaCo 单损失（Cui et al., ICCV 2021；实现与官方 PaCo/LT/losses.py 逐项对齐）——
                    中心 = fc 分类器权重，Balanced Softmax 内嵌于中心项（分母/分子均含 log q 类频率），
                    总损失 L = PaCo(proj(features), labels, fc(features))，无独立 CE、无 w 乘子；
                    训练骨架仍与 joint 一致（warmup/双优化器/best-val/每轮 test）
  spread          : THANOS L_spread（Chen et al., Perfectly Balanced, ICCV 2021）官方实验口径：
                    L = CE + w·[(1-α)·L_sup + α·cNCE]，α=0.5（官方 a_lc=a_spread=1）；
                    L_sup 为 L_in 形式（正例含自身增强对）、cNCE 分母含正例（-log softmax），
                    均以官方代码为准（论文 Eq.2-3 与官方代码有三处不一致，见 SpreadLoss docstring）。
                    按论文要求对训练集加标准对比增强（Lemma 3：去掉增强 cNCE 无意义），
                    本项目唯一带数据增强的臂；每步双前向（原图 + 增强图），耗时约 joint 的 1.8 倍

与文献标准 SupCon 的已知偏离（写实验记录时必须注明）:
  - 不加数据增强：本项目所有臂的训练管线均无增强，SupCon 臂保持一致以保臂间公平
    （唯一例外：spread 臂的 cNCE 正例就是增强视图，必须加增强才能忠实复刻论文）
  - "正样本" = batch 内同标签样本（非同图两视角）

用法（SimEmotion 目录、venv 激活下）:
  python supcon.py --mode joint --num_epochs 25 --seed 43 --datadir FI_seed43
  python supcon.py --mode twostage --stage1_epochs 25 --stage2_epochs 10 --seed 43 --datadir FI_seed43
  python supcon.py --mode pure --num_epochs 25 --knn_k 20 --seed 43 --datadir FI_seed43
  python supcon.py --mode genscl --num_epochs 25 --seed 43 --datadir FI_seed43
  python supcon.py --mode paco --num_epochs 25 --paco_alpha 0.05 --seed 43 --datadir FI_seed43
  python supcon.py --mode spread --num_epochs 25 --spread_alpha 0.7 --seed 43 --datadir FI_seed43

说明:
  - 复用 main.py 的数据/模型/评估逻辑（import main as M 注入模式，同 test_saved_model.py）；
    M.args 必须先于 datasetInit 设置（datasetInit 内部读取 args.datadir）
  - 日志格式与 main.py 完全一致（Train/Val/Test@Epoch/topacc/PerClassF1/混淆矩阵），
    ECC/EMC 可从最终混淆矩阵离线补算
  - checkpoint 命名把 supcon 标记插在数据集名之前（如 modelViT-B-16_supconjoint_FI_8_*.pth），
    不会被 test_saved_model.py 的 modelViT-B-16_FI_8_* 通配误捞；本脚本自带收尾测试
  - batch 32 下每类平均约 4 个正样本对，偏少是 8GB 显存约束下的现实，解读结果时注意
"""
import argparse
import copy
import os
import time
import types

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import main as M


class SupConLoss(nn.Module):
    """SupCon batch 多正样本版：同 batch 同标签互为正样本（无增强两视角）
    L_i = -1/|P(i)| Σ_{p∈P(i)} log( exp(sim(z_i,z_p)/τ) / Σ_{a∈A(i)} exp(sim(z_i,z_a)/τ) )
    输入特征须已 L2 归一化；逐行减 max 保数值稳定；返回标量（batch 均值）"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, features, labels):
        b = features.size(0)
        sim = features @ features.t() / self.tau                      # [B,B] 余弦相似度（已归一化）
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()   # 逐行减 max，数值稳定
        eye = torch.eye(b, dtype=torch.bool, device=features.device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # P(i)：同类且非自身
        log_denom = torch.logsumexp(logits.masked_fill(eye, float('-inf')), dim=1, keepdim=True)  # log Σ_{a∈A(i)}
        log_prob = (logits - log_denom).masked_fill(eye, 0.0)         # [B,B]，对角置 0 不影响求和
        pos_count = pos_mask.sum(1)                                   # |P(i)|
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(1) / pos_count.clamp(min=1)
        valid = pos_count > 0                                         # batch 内无同类的样本不参与均值
        if not valid.any():
            return features.sum() * 0.0                               # 保梯度图的占位 0
        return -mean_log_prob_pos[valid].mean()


class GenSCLLoss(nn.Module):
    """广义监督对比损失（GenSCL, Kim et al. arXiv:2206.00384 Eq.2）忠实复刻：
    L_gen = -1/|A(i)| Σ_{i∈I} Σ_{j∈A(i)} sim(y_i,y_j) · log( exp(z_i·z_j/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) )
    sim(y_i,y_j) = 标签向量的余弦相似度：one-hot 下退化为同类的 0/1 指示阵，
    混合标签（CutMix/KD 软目标）下自动推广为 [0,1] 连续权重（本模式当前只用 one-hot）
    与 SupConLoss 的两点数学差异（复刻对照臂的意义所在）:
      1) 逐 anchor 归一化用 1/|A(i)|（B-1，全部对比样本）而非 1/|P(i)|（正样本数）
      2) 无 valid 掩码——无同类正样本的 anchor 贡献 0 损失行，仍计入均值（与论文公式一致）
    量级提示：one-hot 下 L_gen ≈ SupCon × |P(i)|/(B-1)（batch 32、每类约 4 正样本时约 0.1 倍），
    同 w=0.5 时对比项的实际强度远小于 joint 臂——臂间对比时必须注明，或按量级折算 w
    输入特征须已 L2 归一化；逐行减 max 保数值稳定；返回标量（batch 均值）"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, features, labels):
        b = features.size(0)
        y_onehot = F.one_hot(labels, num_classes=int(labels.max().item()) + 1).float()
        sim_y = y_onehot @ y_onehot.t()                               # [B,B] 标签相似度 ∈ [0,1]，one-hot 范数恒为 1
        logits = features @ features.t() / self.tau                   # [B,B] 余弦相似度/τ（特征已归一化）
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        eye = torch.eye(b, dtype=torch.bool, device=features.device)
        log_denom = torch.logsumexp(logits.masked_fill(eye, float('-inf')), dim=1, keepdim=True)  # log Σ_{a∈A(i)}
        log_prob = (logits - log_denom).masked_fill(eye, 0.0)         # 对角置 0（j∈A(i) 不含自身）
        per_anchor = -(sim_y * log_prob).sum(1) / max(b - 1, 1)       # 1/|A(i)| 归一化，|A(i)|=B-1
        return per_anchor.mean()


class PaCoLoss(nn.Module):
    """官方口径 PaCo 单损失（Cui et al., ICCV 2021；与官方 PaCo/LT/losses.py PaCoLoss 逐项对齐，K=0 batch 版）
    中心 = fc 分类器权重：经 sup_logits = fc(penultimate 未归一化特征) 传入（官方 --normalize 默认 False，
    与本项目评估路径 get_features_test 的 fc(encode_image) 完全一致）；样本对比在 proj 空间（已归一化）。
    L_i = -1/W_i [ β·log(ψ(c_y)/Den) + α·Σ_{p∈P(i)} log(ψ(z_p)/Den) ]
      ψ(c_k) = exp(fc_k(x_i) + log q(y_k))    ← Balanced Softmax 内嵌于中心/分类器项（官方 sup_logits+log(weight)）
      ψ(z_p) = exp(z_p·g_i/τ)                 ← 样本项不加权（官方 γ=1）
      Den   = Σ_k ψ(c_k) + Σ_{z∈A(i)} ψ(z)    （全部中心 + batch 内除自身外全部样本）
      W_i   = β + α·|P(i)|（官方 /mask.sum(1)，β=1）
    总损失 = PaCo 单损失：无独立 CE、无 w 乘子（官方 loss = criterion(features, labels, logits) 单行）
    性质：
      - α=0 时样本正对归零、损失趋近 BS-CE（分母仍含样本项，非严格退化——官方同此行为，已与官方实现对拍验证）
      - 中心正对分子同含 +log q_y（与官方一致；log q 为常数，只平移损失值不影响梯度）
      - 每个 anchor 至少有本类中心一个正对，无孤立样本问题
    与官方的偏离：无 queue/动量编码器/两视角增强（batch 版，同本脚本其他臂口径）；warmup 沿用本脚本骨架
    （warmup 期间 backbone 冻结，fc=中心与 proj 照常训练——分类器从第 0 轮即受 BS 校正）
    输入：features_proj 已 L2 归一化 [B,d]；sup_logits = fc(未归一化 penultimate) [B,C]；返回标量（batch 均值）"""
    def __init__(self, temperature=0.07, alpha=0.05, beta=1.0, gamma=1.0, num_classes=8, q=None):
        super().__init__()
        self.tau = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('log_q', torch.log(q.clamp(min=1e-9)))   # [C] log 类频率，BS 内嵌

    def forward(self, features_proj, labels, sup_logits):
        b = features_proj.size(0)
        n_class = sup_logits.size(1)
        sample_logits = features_proj @ features_proj.t() / self.tau          # [B,B] 样本-样本（proj 空间）
        center_logits = sup_logits + self.log_q.unsqueeze(0)                  # [B,C] BS 内嵌（supt=1）
        logits = torch.cat([center_logits, sample_logits], dim=1)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()     # 逐行减 max，数值稳定
        eye = torch.eye(b, dtype=torch.bool, device=features_proj.device)
        eye_f = eye.float()
        # 分子掩码：中心部分 β·one_hot（γ 无关）；样本部分 α·同类且非自身
        pos_mask = torch.cat([self.beta * F.one_hot(labels, num_classes=n_class).float(),
                              self.alpha * (labels.unsqueeze(0) == labels.unsqueeze(1)).float() * (1.0 - eye_f)],
                             dim=1)
        # 分母掩码：中心全部参与（γ 对中心为 1）；样本除自身外全部参与（权重 γ=1）
        den_mask = torch.cat([torch.ones(b, n_class, device=features_proj.device), 1.0 - eye_f], dim=1)
        log_den = torch.logsumexp(logits.masked_fill(den_mask == 0, float('-inf')), dim=1, keepdim=True)
        log_prob = logits - log_den
        loss_i = -(pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        return loss_i.mean()


class SpreadLoss(nn.Module):
    """官方实验口径 THANOS L_spread（Chen et al., Perfectly Balanced, ICCV 2021）
    数学对齐 code/thanos-code unagi/tasks/loss_fns/contrastive_loss.py 的 ContrastiveLoss(type='l_spread')
    实测配置（官方 repo 用户已拉取；实验配置 cifar10_lspread_cep.yaml 无任何参数覆盖，走默认值）。
    L = (1-α)·L_sup + α·L_cNCE；官方 a_lc=a_spread=1.0 → 等价 α=0.5（默认）
    论文-代码三处不一致（均以官方代码为准）：
      1) α：官方 0.5，落在论文理论 spread 区 α∈(2/3,~0.72) 之外
      2) L_sup 是 L_in 形式：每个正例 j 的分母 = {j 自己} ∪ 全部异类，其他正例不进分母
         （官方 pos_in_denom=False；论文 Eq.2 分母含全部正例 = Khosla L_out）
         且正例集包含"自身增强对"(x_i, a(x_i))——官方跨视图结构的对角正例
         映射到本项目"原图 batch + 单增强视图"结构
      3) L_cNCE 分母包含正例自身（-log softmax 形式，损失有下界 0）
         （论文 Eq.3 分母不含自身；官方 denominator_spread 的 pos_ones 含对角）
    官方结构红利：每个 anchor 至少有自身增强对一个正对，无空正例问题，无需 valid 掩码
      （p_count=0 时 sup 只剩自身对项、cNCE 精确为 0，与官方逐式一致）
    本项目协议偏离（损失数学已对齐官方，训练骨架保持臂间一致，实验记录需注明）:
      - τ=0.07（官方 0.5）、对比项权重 w=0.5（官方 1.0）、有 proj 头（官方无，encoder 特征直接算）
      - sup 正例 = 同类原图 + 自身增强对（官方为双增强视图跨视图互算，无原图参与）
      - CE 无 label smoothing（官方 cep_image_supcon 带 label smoothing）
      - 增强：RRC+flip+ColorJitter+RandomGrayscale 与官方逐项一致（官方多一次冗余 flip），CLIP 归一化
    数值稳定：分子正例恒在自身 logsumexp 集合内，行减 max 的移位自动抵消，损失值精确
    输入：features/features_aug 均须已 L2 归一化 [B,d]（proj 输出）；labels [B]；返回标量"""
    def __init__(self, temperature=0.07, alpha=0.5):
        super().__init__()
        self.tau = temperature
        self.alpha = alpha

    def forward(self, features, labels, features_aug):
        b = features.size(0)
        logits = features @ features.t() / self.tau                   # [B,B] 原图间相似度/τ
        pos_aug = (features * features_aug).sum(1) / self.tau         # [B] 自身增强对
        # 行减同一常数做数值稳定（分子正例在其自身 logsumexp 集合内，移位精确抵消）
        shift = torch.maximum(logits.max(dim=1).values, pos_aug).detach().unsqueeze(1)
        logits = logits - shift
        pos_aug = pos_aug - shift.squeeze(1)
        neginf = float('-inf')
        same = labels.unsqueeze(0) == labels.unsqueeze(1)             # [B,B] 含对角
        neg_mask = ~same                                              # 异类（对角为同类，不在此列）
        other_mask = same & ~torch.eye(b, dtype=torch.bool, device=features.device)   # 同类其他
        log_den_neg = torch.logsumexp(logits.masked_fill(~neg_mask, neginf), dim=1)   # [B]
        # ---- L_sup（L_in 形式）：正例 = 同类其他 + 自身增强对；每个正例分母 = 自己 + 全部异类 ----
        s_pos = logits.masked_fill(~other_mask, 0.0)
        term_pos = (-s_pos + torch.logaddexp(s_pos, log_den_neg.unsqueeze(1)))
        term_pos = term_pos.masked_fill(~other_mask, 0.0)             # 无效位置归零
        p_count = other_mask.sum(1)                                   # 同类其他数（可为 0）
        sup_i = (term_pos.sum(1) + (-pos_aug + torch.logaddexp(pos_aug, log_den_neg))) / (p_count + 1.0)
        # ---- L_cNCE：正例 = 自身增强对；分母 = 正例 + 同类其他（官方含正例本身，-log softmax 形式） ----
        log_den_same = torch.logsumexp(logits.masked_fill(~other_mask, neginf), dim=1)
        cnce_i = -pos_aug + torch.logaddexp(pos_aug, log_den_same)    # p_count=0 时精确为 0
        return (1.0 - self.alpha) * sup_i.mean() + self.alpha * cnce_i.mean()


class AugPairDataset(torch.utils.data.Dataset):
    """spread 模式专用训练集包装：每个样本返回 (原图 CLIP 视图, 增强视图, label)
    增强为对比学习标准轻量组合（SimCLR 式）：RandomResizedCrop + 水平翻转 + 色彩抖动 + 随机灰度，
    最后用 CLIP 的 mean/std 归一化，保证与原图视图同分布口径
    本项目所有臂中唯一带数据增强的训练管线——论文 Lemma 3：cNCE 正例就是增强视图，
    去掉增强该损失项无意义（论文消融：去掉增强掉 35.4 个点）"""
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, base, size=224):
        self.base = base
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self.CLIP_MEAN, self.CLIP_STD),
        ])

    def __getitem__(self, item):
        path, label = self.base.images[item]
        img = Image.open(path).convert('RGB')
        return M.preprocess(img).to(M.device), self.aug(img).to(M.device), self.base.classindex[label]

    def __len__(self):
        return len(self.base)


class SupConProjector(nn.Module):
    """文献标准两层投影头：512 → 512 → 128，输出 L2 归一化（归一化在 forward 内完成）"""
    def __init__(self, dim=512, hidden=512, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=-1, keepdim=True)


@torch.no_grad()
def extract_z(model, proj, dataset, batch_size, device):
    """编码整个数据集，返回投影头输出（已 L2 归一化）的 z [N,128] 与标签 [N]"""
    model.eval()
    zs, ys = [], []
    for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False)):
        features = model.encode_image(images.to(device)).float()
        zs.append(proj(features))
        ys.append(labels)
    model.train()
    return torch.cat(zs), torch.cat(ys)


def knn_classify(train_z, train_y, query_z, k):
    """kNN 分类：余弦相似度加权投票（z 已归一化，内积即余弦）。返回逐查询预测 [N]"""
    preds = []
    train_y = train_y.to(train_z.device)                           # 标签搬到与 z 同设备，避免跨设备索引报错
    n_class = int(train_y.max().item()) + 1
    k = min(k, train_z.size(0))                                    # 防御：k 不得超过训练集大小
    for i in range(0, query_z.size(0), 1024):                      # 分块防大矩阵爆显存
        sim = query_z[i:i + 1024] @ train_z.t()                    # [chunk, N_train]
        topv, topi = sim.topk(k, dim=1)                            # 最近 k 个训练样本
        votes = torch.zeros(topi.size(0), n_class, device=query_z.device)
        for c in range(n_class):
            votes[:, c] = (topv * (train_y[topi] == c)).sum(1)     # 同类邻居相似度求和=加权票
        preds.append(votes.argmax(1))
    return torch.cat(preds)


def knn_evaluate(model, proj, trainset, query_dataset, classtype, batch_size, device, k, log, prefix, train_feats=None):
    """kNN 评估：从混淆矩阵算 PerClassF1/macroF1，日志格式与 get_features_test 对齐。
    train_feats 可传入已编码的训练集 (z, y) 以避免同一模型下重复编码（每轮 val+test 共用）"""
    if train_feats is None:
        train_feats = extract_z(model, proj, trainset, batch_size, device)
    train_z, train_y = train_feats
    query_z, query_y = extract_z(model, proj, query_dataset, batch_size, device)
    preds = knn_classify(train_z, train_y, query_z, k)
    truth = query_y.to(preds.device)
    k_len = len(classtype)
    testResult = [[0] * k_len for _ in range(k_len)]
    for t, p in zip(truth.cpu().tolist(), preds.cpu().tolist()):
        testResult[t][p] += 1
    correct = sum(testResult[i][i] for i in range(k_len))
    acc = correct / len(truth)
    f1s, macro_f1 = M.f1_from_matrix(testResult)
    log.append('PerClassF1: ' + str([round(f, 4) for f in f1s]) + '\n')
    log.append(str(testResult) + '\n')
    info = prefix + ' kNN(k={}) Acc: {:.4f} macroF1: {:.4f}'.format(k, acc, macro_f1)
    log.append(info + '\n')
    print(info)
    print('Answers in testlist:', testResult)
    return acc


def run_val(model, fc, criterion, valdataset, batch_size, device):
    """内联验证循环（main.py 同款格式：返回 loss/acc，不打印混淆矩阵）"""
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(valdataset, batch_size=batch_size, shuffle=False)):
            images = images.to(device)
            labels = labels.to(device)
            features = model.encode_image(images).float()
            outputs = fc(features)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(valdataset)
    epoch_acc = running_corrects.double() / len(valdataset)
    return epoch_loss, epoch_acc


def save_checkpoints(model, fc, proj, tag, datasetname, ts, save_fc=True):
    """supcon 标记插在数据集名之前，避免被 test_saved_model.py 的 modelViT-B-16_FI_8_* 通配误捞；
    pure 模式 fc 从未训练，不存（save_fc=False）以免误导；
    paco 模式中心=fc 权重，fc 三件套已覆盖，无需额外中心文件"""
    os.makedirs('./savemodel', exist_ok=True)
    torch.save(model.state_dict(), f'./savemodel/model{tag}_{datasetname}_{ts}.pth')
    if save_fc:
        torch.save(fc.state_dict(), f'./savemodel/fc{tag}_{datasetname}_{ts}.pth')
    torch.save(proj.state_dict(), f'./savemodel/proj{tag}_{datasetname}_{ts}.pth')


def train_joint(args, classtype, device, trainset, valset, testset, model, fc, proj, log):
    """联合模式：L = CE + w·SupCon 单循环，骨架与 main.py get_features 一致；
    genscl 模式复用同一骨架，仅把对比项从 SupConLoss 换成 GenSCLLoss（Eq.2 复刻）；
    paco 模式为官方口径单损失：L = PaCo(proj(features), labels, fc(features))，无独立 CE、无 w
    （中心=fc 权重、BS 内嵌；PaCoLoss 无可学习参数，fc 仍由 optimizer_fc 按原口径优化）；
    spread 模式（THANOS L_spread 官方口径）：L = CE + w·[(1-α)·L_sup + α·cNCE]，训练集经 AugPairDataset
    包装双视图加载，增强图进对比项（L_sup 自身对 + cNCE 正例），CE 头只看原图，评估管线与其他臂完全一致"""
    criterion = nn.CrossEntropyLoss()
    paco = None
    spread_criterion = None
    if args.mode == 'genscl':
        supcon_criterion = GenSCLLoss(temperature=args.temperature)
    elif args.mode == 'paco':
        # 类频率 q(y) = n_c/N（与 main.py wce 哨兵臂同口径），Balanced Softmax 内嵌于中心=fc 项
        counts = [sum(1 for _, l in trainset.images if l == c) for c in classtype]
        q = torch.tensor([n / max(sum(counts), 1) for n in counts], dtype=torch.float32)
        paco = PaCoLoss(temperature=args.temperature, alpha=args.paco_alpha,
                        num_classes=len(classtype), q=q).to(device)
        supcon_criterion = paco
    elif args.mode == 'spread':
        spread_criterion = SpreadLoss(temperature=args.temperature, alpha=args.spread_alpha).to(device)
        supcon_criterion = None
    else:
        supcon_criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=1e-5)
    optimizer_fc = torch.optim.Adam(fc.parameters(), lr=args.lr_FC)
    optimizer_proj = torch.optim.Adam(proj.parameters(), lr=args.lr_FC)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize_BB, gamma=args.gamma_BB)
    scheduler_fc = lr_scheduler.StepLR(optimizer_fc, step_size=args.stepsize_FC, gamma=args.gamma_FC)
    scheduler_proj = lr_scheduler.StepLR(optimizer_proj, step_size=args.stepsize_FC, gamma=args.gamma_FC)

    topacc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    best_fc = copy.deepcopy(fc.state_dict())
    best_proj = copy.deepcopy(proj.state_dict())

    # spread 模式用双视图数据集（原图 + 增强图），其余臂用原训练集
    loader_src = AugPairDataset(trainset) if args.mode == 'spread' else trainset

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for batch in tqdm(DataLoader(loader_src, batch_size=args.batch_size, shuffle=True)):
            if args.mode == 'spread':
                images, images_aug, labels = batch
                images_aug = images_aug.to(device)
            else:
                images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            if epoch >= args.warmepoch:
                optimizer.zero_grad()
            optimizer_fc.zero_grad()
            optimizer_proj.zero_grad()
            with torch.set_grad_enabled(True):
                features = model.encode_image(images).float()
                outputs = fc(features)
                if args.mode == 'paco':
                    # 官方口径 PaCo 单损失：中心=fc 权重、BS 内嵌，无独立 CE、无 w
                    loss = supcon_criterion(proj(features), labels, outputs)
                elif args.mode == 'spread':
                    # THANOS L_spread 官方口径：增强图进对比项（L_sup 自身对 + cNCE 正例），CE 头只看原图
                    features_aug = model.encode_image(images_aug).float()
                    spread = spread_criterion(proj(features), labels, proj(features_aug))
                    loss = criterion(outputs, labels) + args.supcon_weight * spread
                else:
                    ce = criterion(outputs, labels)
                    sup = supcon_criterion(proj(features), labels)
                    loss = ce + args.supcon_weight * sup
                _, preds = torch.max(outputs, 1)
                loss.backward()
                if epoch >= args.warmepoch:
                    M.convert_models_to_fp32(model.visual)
                    optimizer.step()
                    clip.model.convert_weights(model.visual)
                optimizer_fc.step()
                optimizer_proj.step()
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_losstrain = running_loss / len(trainset)
        epoch_acctrain = running_corrects.double() / len(trainset)
        loginfo = 'Train Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_losstrain, epoch_acctrain)
        log.append(loginfo + '\n')
        print(loginfo)

        if epoch >= args.warmepoch:
            scheduler.step()
        scheduler_fc.step()
        scheduler_proj.step()

        epoch_lossval, epoch_accval = run_val(model, fc, criterion, valset, args.batch_size, device)
        if epoch_accval > topacc:
            topacc = epoch_accval
            best_model = copy.deepcopy(model.state_dict())
            best_fc = copy.deepcopy(fc.state_dict())
            best_proj = copy.deepcopy(proj.state_dict())
        loginfo = 'Val Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_lossval, epoch_accval)
        log.append(loginfo + '\n')
        log.append('topacc: ' + str(topacc) + '\n')
        print(loginfo)
        print('topacc: ' + str(topacc))
        # 每轮用当前模型在测试集上评估一次（仅记录曲线，模型选择仍以 val 为准）
        testacc = M.get_features_test(testset)
        epoctestinfo = 'Test@Epoch:{} Acc: {:.4f}'.format(epoch, testacc)
        log.append(epoctestinfo + '\n')
        print(epoctestinfo)

    model.load_state_dict(best_model)
    fc.load_state_dict(best_fc)
    proj.load_state_dict(best_proj)
    M.get_features_test(testset)   # best-val 回载后的最终测试（混淆矩阵/PerClassF1/macroF1 进日志）


def train_twostage(args, classtype, device, trainset, valset, testset, model, fc, proj, log):
    """两阶段模式：Stage1 SupCon 预训练 backbone+proj → Stage2 冻结特征 CE 训 fc"""
    supcon_criterion = SupConLoss(temperature=args.temperature)
    criterion = nn.CrossEntropyLoss()

    # ---- Stage1: SupCon 预训练（无 CE 无 fc，无 warmup——SupCon 直接塑形特征）----
    # --stage1_ckpt 传入时间戳时跳过训练，直接加载已完成的 SupCon 预训练权重
    # （如 pure 模式产物；训练循环与 Stage1 数学上等价，省 25 轮 ~1.8h。
    #   注意：pure 存的是 best-val kNN 权重，与朴素 twostage 的"末轮权重"略不同，属协议偏差需注明）
    if args.stage1_ckpt:
        import glob
        hits_m = glob.glob('./savemodel/model*_' + args.datasetname + '_' + args.stage1_ckpt + '.pth')
        hits_p = glob.glob('./savemodel/proj*_' + args.datasetname + '_' + args.stage1_ckpt + '.pth')
        if not hits_m or not hits_p:
            raise FileNotFoundError(
                'Stage1 checkpoint not found: need model*/proj* pair for dataset '
                + args.datasetname + ' ts=' + args.stage1_ckpt)
        model.load_state_dict(torch.load(hits_m[0], map_location='cpu'))
        proj.load_state_dict(torch.load(hits_p[0], map_location='cpu'))
        loginfo = 'Stage1 loaded from checkpoint: ' + hits_m[0] + ' (skip training)'
        log.append(loginfo + '\n')
        print(loginfo)
    else:
        optimizer = torch.optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=1e-5)
        optimizer_proj = torch.optim.Adam(proj.parameters(), lr=args.lr_FC)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize_BB, gamma=args.gamma_BB)
        scheduler_proj = lr_scheduler.StepLR(optimizer_proj, step_size=args.stepsize_FC, gamma=args.gamma_FC)

        for epoch in range(args.stage1_epochs):
            running_loss = 0.0
            for images, labels in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True)):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                optimizer_proj.zero_grad()
                with torch.set_grad_enabled(True):
                    features = model.encode_image(images).float()
                    loss = supcon_criterion(proj(features), labels)
                    loss.backward()
                    M.convert_models_to_fp32(model.visual)
                    optimizer.step()
                    clip.model.convert_weights(model.visual)
                    optimizer_proj.step()
                running_loss += loss.item() * features.size(0)
            loginfo = 'Stage1 SupCon Epoch:{} Loss: {:.4f}'.format(epoch, running_loss / len(trainset))
            log.append(loginfo + '\n')
            print(loginfo)
            scheduler.step()
            scheduler_proj.step()

    # ---- Stage2: backbone 冻结，纯 CE 训 fc（文献标准）----
    model.eval()
    for p in model.visual.parameters():
        p.requires_grad = False
    optimizer_fc = torch.optim.Adam(fc.parameters(), lr=args.lr_FC)
    scheduler_fc = lr_scheduler.StepLR(optimizer_fc, step_size=args.stepsize_FC, gamma=args.gamma_FC)

    topacc = 0.0
    best_fc = copy.deepcopy(fc.state_dict())
    for epoch in range(args.stage2_epochs):
        running_loss = 0.0
        running_corrects = 0
        for images, labels in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer_fc.zero_grad()
            with torch.set_grad_enabled(True):
                with torch.no_grad():                      # backbone 冻结：编码不建图
                    features = model.encode_image(images).float()
                outputs = fc(features)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer_fc.step()
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_losstrain = running_loss / len(trainset)
        epoch_acctrain = running_corrects.double() / len(trainset)
        loginfo = 'Stage2 CE Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_losstrain, epoch_acctrain)
        log.append(loginfo + '\n')
        print(loginfo)
        scheduler_fc.step()

        # 用 M.get_features_test 评估 val（重编码开销可接受，换日志格式完全统一；prefix='Val' 区分 test 行）
        epoch_accval = M.get_features_test(valset, prefix='Val')
        if epoch_accval > topacc:
            topacc = epoch_accval
            best_fc = copy.deepcopy(fc.state_dict())
        log.append('Stage2 topacc: ' + str(topacc) + '\n')
        print('Stage2 topacc: ' + str(topacc))
        # 每轮用当前模型在测试集上评估一次（仅记录曲线，模型选择仍以 val 为准，与 joint/pure 口径一致）
        testacc = M.get_features_test(testset)
        epoctestinfo = 'Stage2 Test@Epoch:{} Acc: {:.4f}'.format(epoch, testacc)
        log.append(epoctestinfo + '\n')
        print(epoctestinfo)

    fc.load_state_dict(best_fc)
    M.get_features_test(testset)   # best-val 回载后的最终测试


def train_pure(args, classtype, device, trainset, valset, testset, model, proj, log):
    """纯 SupCon 模式：全程无 CE 梯度，kNN 评估。训练循环同 twostage Stage1，
    每轮 val+test kNN 双评估（训练集编码一次共用），best-val 选择只看 val（test 仅观察）"""
    supcon_criterion = SupConLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=1e-5)
    optimizer_proj = torch.optim.Adam(proj.parameters(), lr=args.lr_FC)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize_BB, gamma=args.gamma_BB)
    scheduler_proj = lr_scheduler.StepLR(optimizer_proj, step_size=args.stepsize_FC, gamma=args.gamma_FC)

    topacc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    best_proj = copy.deepcopy(proj.state_dict())

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True)):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimizer_proj.zero_grad()
            with torch.set_grad_enabled(True):
                features = model.encode_image(images).float()
                loss = supcon_criterion(proj(features), labels)
                loss.backward()
                M.convert_models_to_fp32(model.visual)
                optimizer.step()
                clip.model.convert_weights(model.visual)
                optimizer_proj.step()
            running_loss += loss.item() * features.size(0)
        loginfo = 'SupCon Epoch:{} Loss: {:.4f}'.format(epoch, running_loss / len(trainset))
        log.append(loginfo + '\n')
        print(loginfo)
        scheduler.step()
        scheduler_proj.step()

        # 每轮 val + test kNN：训练集只编码一次共用；test 仅观察不参与 checkpoint 选择（与 main.py 七档臂口径一致）
        train_feats = extract_z(model, proj, trainset, args.batch_size, device)
        valacc = knn_evaluate(model, proj, trainset, valset, classtype,
                              args.batch_size, device, args.knn_k, log,
                              prefix='Val@Epoch:{}'.format(epoch), train_feats=train_feats)
        if valacc > topacc:
            topacc = valacc
            best_model = copy.deepcopy(model.state_dict())
            best_proj = copy.deepcopy(proj.state_dict())
        log.append('topacc: ' + str(topacc) + '\n')
        print('topacc: ' + str(topacc))
        knn_evaluate(model, proj, trainset, testset, classtype,
                     args.batch_size, device, args.knn_k, log,
                     prefix='Test@Epoch:{}'.format(epoch), train_feats=train_feats)

    model.load_state_dict(best_model)
    proj.load_state_dict(best_proj)
    knn_evaluate(model, proj, trainset, testset, classtype,
                 args.batch_size, device, args.knn_k, log, prefix='FinalTest')


def main():
    parser = argparse.ArgumentParser(description='Supervised Contrastive Loss experiments for SimEmotion')
    parser.add_argument('--datasetname', type=str, default='FI_8')
    parser.add_argument('--modelname', type=str, default='ViT-B/16', help='仅支持 ViT-B/16（fcdim=512）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--mode', type=str, default='joint', choices=['joint', 'twostage', 'pure', 'genscl', 'paco', 'spread'],
                        help='joint=CE+w·SupCon 联合单循环；twostage=SupCon 预训练+CE 两阶段；'
                             'pure=纯 SupCon+kNN 评估；genscl=CE+w·GenSCL（Eq.2 复刻，骨架同 joint）；'
                             'paco=官方口径 PaCo 单损失（中心=fc 权重+BS 内嵌，无独立 CE）；'
                             'spread=THANOS L_spread（CE+w·[(1-α)SupCon+α·cNCE]，唯一带增强的臂）')
    parser.add_argument('--knn_k', type=int, default=20, help='pure 模式 kNN 的邻居数 k')
    parser.add_argument('--num_epochs', type=int, default=25, help='joint/pure 模式总轮数')
    parser.add_argument('--stage1_epochs', type=int, default=25, help='twostage 阶段一（SupCon 预训练）轮数')
    parser.add_argument('--stage2_epochs', type=int, default=10, help='twostage 阶段二（CE 训 fc）轮数')
    parser.add_argument('--stage1_ckpt', type=str, default=None,
                        help='twostage 跳过 Stage1：加载已有 SupCon 预训练 checkpoint（传时间戳如 20261025093610，'
                             '需存在 model*/proj* 同戳两件套；须与当前 seed/数据划分一致')
    parser.add_argument('--temperature', type=float, default=0.07, help='SupCon 温度 τ')
    parser.add_argument('--supcon_weight', type=float, default=0.5,
                        help='joint/genscl 模式对比项权重 w（paco 为单损失，此参数不生效仅回显）')
    parser.add_argument('--paco_alpha', type=float, default=0.05,
                        help='paco 模式样本正对权重 α（论文正文 0.05，官方脚本 default 1.0；β=γ=1 固定）')
    parser.add_argument('--spread_alpha', type=float, default=0.5,
                        help='spread 模式 cNCE 权重 α：L=(1-α)·L_sup+α·cNCE（官方实验配置 a_lc=a_spread=1 → α=0.5；'
                             '论文理论 spread 区 α>2/3 与官方配置不一致，按官方代码为准；α=0 退化为纯 L_in 版 SupCon）')
    parser.add_argument('--lr_BB', type=float, default=1e-6)
    parser.add_argument('--lr_FC', type=float, default=1e-3)
    parser.add_argument('--stepsize_BB', type=int, default=6)
    parser.add_argument('--gamma_BB', type=float, default=0.5)
    parser.add_argument('--stepsize_FC', type=int, default=6)
    parser.add_argument('--gamma_FC', type=float, default=0.5)
    parser.add_argument('--warmepoch', type=int, default=3, help='joint 模式 warmup 轮数（backbone 冻结）')
    parser.add_argument('--datadir', type=str, default=None, help='数据划分目录（如 FI_seed43），结尾斜杠可省略')
    args = parser.parse_args()
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'   # 路径按字符串拼接，结尾斜杠必须补齐

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 注入 main 模块全局量（顺序关键：args 先于 datasetInit）----
    M.args = types.SimpleNamespace(datasetname=args.datasetname, batch_size=args.batch_size, datadir=args.datadir)
    classtype, data_dir = M.datasetInit(args.datasetname)
    M.classtype = classtype
    M.data_dir = data_dir
    M.device = device
    M.criterion = nn.CrossEntropyLoss()   # get_features_test 的 loss 日志口径
    log = []
    M.log = log
    M.set_seed(args.seed)

    model, preprocess = clip.load(args.modelname, device)
    clip.model.convert_weights(model)
    M.preprocess = preprocess
    fc = M.FCN(dim=512).to(device)
    proj = SupConProjector(dim=512).to(device)
    M.model = model
    M.fc = fc
    args.modelname = args.modelname.replace('/', '-')

    trainset = M.Loaddata('train')
    valset = M.Loaddata('val')
    testset = M.Loaddata('test')

    print("New SupCon experiment start.")
    print('mode: ' + args.mode + ' tau: ' + str(args.temperature) + ' supcon_weight: ' + str(args.supcon_weight)
          + ' paco_alpha: ' + str(args.paco_alpha) + ' spread_alpha: ' + str(args.spread_alpha))
    starttime = time.strftime('%Y%m%d%H%M%S')
    print(starttime)
    startpoint = time.time()

    if args.mode in ('joint', 'genscl', 'paco', 'spread'):
        train_joint(args, classtype, device, trainset, valset, testset, model, fc, proj, log)
        tag = {'joint': 'supconjoint', 'genscl': 'supcongenscl', 'paco': 'supconpaco',
               'spread': 'supconspread'}[args.mode]
    elif args.mode == 'twostage':
        train_twostage(args, classtype, device, trainset, valset, testset, model, fc, proj, log)
        tag = 'supcon2stage'
    else:
        train_pure(args, classtype, device, trainset, valset, testset, model, proj, log)
        tag = 'supconpure'

    endpoint = time.time()
    print('cost: ' + str(endpoint - startpoint) + 's.')

    # ---- 存 checkpoint（pure 模式 fc 未训练不存；paco 中心=fc 权重，三件套已覆盖）+ 日志 ----
    save_checkpoints(model, fc, proj, tag, args.datasetname, starttime, save_fc=(args.mode != 'pure'))
    paraminfo = args.modelname + '_' + args.datasetname + ' loss:supcon_' + args.mode \
                + ' tau:' + str(args.temperature) + ' w:' + str(args.supcon_weight) \
                + ' paco_alpha:' + str(args.paco_alpha) \
                + ' spread_alpha:' + str(args.spread_alpha) \
                + ' knn_k:' + str(args.knn_k) \
                + ' lr_FC:' + str(args.lr_FC) + ' stepsize_FC:' + str(args.stepsize_FC) \
                + ' gamma_FC' + str(args.gamma_FC) + ' lr_BB:' + str(args.lr_BB) \
                + ' stepsize_BB:' + str(args.stepsize_BB) + ' gamma_BB' + str(args.gamma_BB) \
                + ' warmepoch:' + str(args.warmepoch) + ' seed:' + str(args.seed) \
                + ' datadir:' + str(args.datadir)
    print(paraminfo)
    log.append(paraminfo + '\n')
    os.makedirs('./log', exist_ok=True)
    with open('./log/' + tag + '_' + args.datasetname + '_' + str(starttime) + '.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')


if __name__ == '__main__':
    main()
