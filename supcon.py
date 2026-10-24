"""supcon.py — 监督对比损失（SupCon, Khosla et al., NeurIPS 2020）实验脚本

两种范式（--mode 选择）:
  joint    (默认) : L = CE(fc(features)) + w·SupCon(proj(features))，单循环，
                    与 main.py 七档 loss 臂完全同构（同 warmup/双优化器/调度/best-val），消融对照最公平
  twostage        : 文献标准两阶段——Stage1 SupCon 预训练 backbone+投影头，Stage2 冻结特征用 CE 训分类头

与文献标准 SupCon 的已知偏离（写实验记录时必须注明）:
  - 不加数据增强：本项目所有臂的训练管线均无增强，SupCon 臂保持一致以保臂间公平
  - "正样本" = batch 内同标签样本（非同图两视角）

用法（SimEmotion 目录、venv 激活下）:
  python supcon.py --mode joint --num_epochs 25 --seed 43 --datadir FI_seed43
  python supcon.py --mode twostage --stage1_epochs 25 --stage2_epochs 10 --seed 43 --datadir FI_seed43

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


def save_checkpoints(model, fc, proj, tag, datasetname, ts):
    """supcon 标记插在数据集名之前，避免被 test_saved_model.py 的 modelViT-B-16_FI_8_* 通配误捞"""
    os.makedirs('./savemodel', exist_ok=True)
    torch.save(model.state_dict(), f'./savemodel/model{tag}_{datasetname}_{ts}.pth')
    torch.save(fc.state_dict(), f'./savemodel/fc{tag}_{datasetname}_{ts}.pth')
    torch.save(proj.state_dict(), f'./savemodel/proj{tag}_{datasetname}_{ts}.pth')


def train_joint(args, classtype, device, trainset, valset, testset, model, fc, proj, log):
    """联合模式：L = CE + w·SupCon 单循环，骨架与 main.py get_features 一致"""
    criterion = nn.CrossEntropyLoss()
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

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_corrects = 0
        for images, labels in tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True)):
            images = images.to(device)
            labels = labels.to(device)
            if epoch >= args.warmepoch:
                optimizer.zero_grad()
            optimizer_fc.zero_grad()
            optimizer_proj.zero_grad()
            with torch.set_grad_enabled(True):
                features = model.encode_image(images).float()
                outputs = fc(features)
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

        # 用 M.get_features_test 评估 val（重编码开销可接受，换日志格式完全统一）
        epoch_accval = M.get_features_test(valset)
        if epoch_accval > topacc:
            topacc = epoch_accval
            best_fc = copy.deepcopy(fc.state_dict())
        log.append('Stage2 topacc: ' + str(topacc) + '\n')
        print('Stage2 topacc: ' + str(topacc))

    fc.load_state_dict(best_fc)
    M.get_features_test(testset)   # best-val 回载后的最终测试


def main():
    parser = argparse.ArgumentParser(description='Supervised Contrastive Loss experiments for SimEmotion')
    parser.add_argument('--datasetname', type=str, default='FI_8')
    parser.add_argument('--modelname', type=str, default='ViT-B/16', help='仅支持 ViT-B/16（fcdim=512）')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--mode', type=str, default='joint', choices=['joint', 'twostage'],
                        help='joint=CE+w·SupCon 联合单循环；twostage=SupCon 预训练+CE 两阶段')
    parser.add_argument('--num_epochs', type=int, default=25, help='joint 模式总轮数')
    parser.add_argument('--stage1_epochs', type=int, default=25, help='twostage 阶段一（SupCon 预训练）轮数')
    parser.add_argument('--stage2_epochs', type=int, default=10, help='twostage 阶段二（CE 训 fc）轮数')
    parser.add_argument('--temperature', type=float, default=0.07, help='SupCon 温度 τ')
    parser.add_argument('--supcon_weight', type=float, default=0.5, help='joint 模式 SupCon 项权重 w')
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
    print('mode: ' + args.mode + ' tau: ' + str(args.temperature) + ' supcon_weight: ' + str(args.supcon_weight))
    starttime = time.strftime('%Y%m%d%H%M%S')
    print(starttime)
    startpoint = time.time()

    if args.mode == 'joint':
        train_joint(args, classtype, device, trainset, valset, testset, model, fc, proj, log)
        tag = 'supconjoint'
    else:
        train_twostage(args, classtype, device, trainset, valset, testset, model, fc, proj, log)
        tag = 'supcon2stage'

    endpoint = time.time()
    print('cost: ' + str(endpoint - startpoint) + 's.')

    # ---- 存 checkpoint（三件套）+ 日志 ----
    save_checkpoints(model, fc, proj, tag, args.datasetname, starttime)
    paraminfo = args.modelname + '_' + args.datasetname + ' loss:supcon_' + args.mode \
                + ' tau:' + str(args.temperature) + ' w:' + str(args.supcon_weight) \
                + ' lr_FC:' + str(args.lr_FC) + ' stepsize_FC:' + str(args.stepsize_FC) \
                + ' gamma_FC' + str(args.gamma_FC) + ' lr_BB:' + str(args.lr_BB) \
                + ' stepsize_BB:' + str(args.stepsize_BB) + ' gamma_BB' + str(args.gamma_BB) \
                + ' warmepoch:' + str(args.warmepoch) + ' seed:' + str(args.seed) \
                + ' datadir:' + str(args.datadir)
    print(paraminfo)
    log.append(paraminfo + '\n')
    os.makedirs('./log', exist_ok=True)
    with open('./log/supcon' + ('joint' if args.mode == 'joint' else '2stage') + '_' + args.datasetname + '_' + str(starttime) + '.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')


if __name__ == '__main__':
    main()
