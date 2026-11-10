import os
import clip
import torch
import torch.nn.functional as F
import random
from PIL import ImageFile, Image
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import time
import copy
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True              #允许读取截断的图片


class InputException(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        print('Got a wrong input for ' + self.name + '!')


def datasetInit(datasetname):
    if datasetname == 'FI_8':
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/FI/'
    elif datasetname == 'ArtPhoto':
        # 8 类与 FI_8 同 taxonomy；prepare_artphoto.py 已把 sad 映射为 sadness
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/artphoto_split/'
    elif datasetname == 'ArtPhoto5fold':
        # 官方 artphoto_partition 5-fold（806 张，8 类与 FI_8 同 taxonomy，sad→sadness 已由
        # prepare_artphoto_5fold.py 重命名）；artphoto_5fold/fold{i}/{train,test}/<8 类>/ 已 100%
        # 校验与 config 一致（2026-11-07）。每折无 val → 走官方 val=test 协议（与 Twitter1/Abstract7
        # 同款，见下方强制列表），FinalTest 即 test 峰值口径；与 ArtPhoto（单划分 best-val）不可混排。
        # 名字带 5fold 是有意的：main.py 日志名为 <model>_<datasetname><acc>_，若叫 ArtPhoto5，
        # 通配 ArtPhoto5* 会误匹配旧 ArtPhoto 日志（如 ArtPhoto51.85 = acc 1.85）
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/artphoto_5fold/fold1/'
    elif datasetname == 'EmotionROI':
        # 6 类；joy/surprise 不在 Mikels 8 类 → 仅支持 ce/fal（ListMLE 系与 dist_fal 会报错）
        _classtype = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/emotionroi_split/'
    elif datasetname == 'Abstract':
        # 8 类与 FI_8 同 taxonomy；prepare_abstract.py 已按 CSV 投票 argmax + seed43 生成 abstract_split/
        # （51 张并列图随机分配见 abstract_split_manifest.txt；anger 全集仅 6 张 → per-class anger 无意义；
        #   test 56 张，噪声带 ±5-6pp，结论定性为主并需同时报 test peak）
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/abstract_split/'
    elif datasetname == 'Abstract7':
        # 官方 abstract_partition 5-fold（226 张，7 类无 anger——官方划分本身不含 anger）；
        # abstract_5fold/fold{i}/{train,test}/<7 类>/ 已 100% 校验与 config 一致（2026-11-07）
        # 注意类名是 'sad'（非 sadness）；每折无 val → 必须配 --no_val；与 Abstract（280 张 8 类）不同协议与标签空间，不可混排
        # MASP 可比协议（其 Abstract 数字疑为此 226/7 类口径，见论文数据集清单.md 备注）
        _classtype = ['amusement', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sad']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/abstract_5fold/fold1/'
    elif datasetname == 'Twitter1':
        # 2 类（SimEmotion Twitter I：pos 769 / neg 500，1:1.54）；官方 5-fold（config twitter1269，
        # label 0=neg / 1=pos 已由文件夹归属 100% 验证）；折目录 twitter1_fold{i}/{train,test}/{pos,neg}/
        # 由 prepare_twitter1.py 生成
        # **官方协议**（code/SimEmotion-main/main.py L341-344）：非 FI 数据集 val=test——checkpoint 按
        # test 选择，报告数字即 test 峰值口径；据此 Twitter1 禁用 --no_val（强制 False）
        # classtype ['neg','pos'] 使 neg=0 / pos=1 与官方标签一致；ListMLE 系仅支持 8 类 Mikels，二分类不可用
        _classtype = ['neg', 'pos']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/twitter1_fold1/'
    elif datasetname == 'Twitter2':
        # 2 类（SimEmotion Twitter II：pos 470 / neg 133 = 603 张，1:3.53 不平衡强于 Twitter I）；
        # 官方 5-fold（config twitter603：train_{i}.txt 带标签 + partition{i}.txt test 列表）；
        # 折目录 twitter2_fold{i}/{train,test}/{pos,neg}/ 由 prepare_twitter2.py 生成（GT/train/test
        # 三重校验 100%）；官方每折仅覆盖 587/603 张（16 张不进任何折，官方划分原样）
        # **官方协议**：非 FI 数据集 val=test（与 Twitter1 同款，强制禁用 --no_val）
        # classtype ['neg','pos'] 使 neg=0 / pos=1 与官方标签一致；ListMLE 系不可用
        _classtype = ['neg', 'pos']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/twitter2_fold1/'
    elif datasetname == 'EmoSet':
        # 8 类 Mikels 同 FI taxonomy（EmoSet-118K：train 94,481 / val 5,905 / test 17,716 = 80/5/15）；
        # 官方划分清单 emoset_split_json/{train,val,test}.json（2026-11-10 经 Dropbox Range 方案从
        # EmoSet-118K.zip 精确拉取，与本地 dataset/Emoset 118,102 张 100% 校验——零缺失零标签冲突）；
        # 折目录 emoset_split/{train,val,test}/<小写类名>/ 由 prepare_emoset.py 生成（三划分互斥）
        # 有 val/ → 走标准 best-val 协议（与 FI 官方划分同款）；八类均衡（BS 校正无操作）
        # 8 类全小写与 FI_8 classtype 字母序一致（amusement,anger,awe,contentment,disgust,excitement,fear,sadness）
        # 训练成本：94k train ≈ FI 5 倍 → 25ep 全量 ~10h/臂（建议先 2ep 冒烟测吞吐）
        _classtype = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
        _data_dir = args.datadir or 'D:/Coding/Python_Projects/SimEmotion/emoset_split/'
    else:
        raise InputException
    return _classtype, _data_dir


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    os.environ['PYTHONHASHSEED'] = str(seed)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class Loaddata(torch.utils.data.Dataset):
    def __init__(self, setname):
        self.images = []
        self.setname = setname
        path = data_dir
        self.classindex = {c: i for i, c in enumerate(classtype)}   # 动态适配各数据集的类别表

        for ct in classtype:
            filepath = path + setname + '/' + ct + '/'
            file_list = os.listdir(filepath)
            for filename in file_list:
                img_path = filepath + filename
                img_label = ct
                self.images.append((img_path, img_label))

    def __getitem__(self, item):
        image, label = self.images[item]
        # --aug 时仅训练集走论文增强管线，val/test 仍用确定性 eval preprocess；
        # globals().get 兼容 test_saved_model.py 注入命名空间（该脚本只取 test 集）
        tfm = preprocess
        if self.setname == 'train' and getattr(args, 'aug', False):
            tp = globals().get('train_preprocess')
            if tp is not None:
                tfm = tp
        img = tfm(Image.open(image)).to(device)
        labelindex = self.classindex[label]
        return img, labelindex

    def __len__(self):
        return len(self.images)


def get_features(traindataset, valdataset):
    topacc = 0.0
    best_model = copy.deepcopy(model.state_dict())
    best_fc = copy.deepcopy(fc.state_dict())
    for epoch in range(args.num_epochs):
        running_loss = 0.0
        running_corrects = 0

        for images, labels in tqdm(DataLoader(traindataset, batch_size=args.batch_size, shuffle=True)):
            images = images.to(device)
            labels = labels.to(device)
            if epoch >= args.warmepoch and args.loss != 'vpt':
                optimizer.zero_grad()
            optimizer_fc.zero_grad()
            with torch.set_grad_enabled(True):
                features = encode_features(images)
                outputs = fc(features)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                if epoch >= args.warmepoch and args.loss != 'vpt':
                    # VPT 臂跳过：backbone 全程冻结无可步进参数（convert_models_to_fp32 会因 grad=None 崩溃）；
                    # P token 梯度已随 backward 回传，由 optimizer_fc.step() 统一更新
                    convert_models_to_fp32(model.visual)
                    optimizer.step()
                    clip.model.convert_weights(model.visual)
                optimizer_fc.step()
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_losstrain = running_loss / len(traindataset)
        epoch_acctrain = running_corrects.double() / len(traindataset)
        loginfo = 'Train Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_losstrain, epoch_acctrain)
        log.append(loginfo + '\n')
        print(loginfo)

        if epoch >= args.warmepoch and args.loss != 'vpt':
            scheduler.step()
        scheduler_fc.step()
        if valdataset is not None:
            running_lossval = 0.0
            running_correctsval = 0

            for images, labels in tqdm(DataLoader(valdataset, batch_size=args.batch_size, shuffle=False)):
                images = images.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(False):
                    features = encode_features(images)
                    outputs = fc(features)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                running_lossval += loss.item() * features.size(0)
                running_correctsval += torch.sum(preds == labels.data)
            epoch_lossval = running_lossval / len(valdataset)
            epoch_accval = running_correctsval.double() / len(valdataset)
            if epoch_accval > topacc:
                topacc = epoch_accval
                best_model = copy.deepcopy(model.state_dict())
                best_fc = copy.deepcopy(fc.state_dict())

            loginfo = 'Val Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_lossval, epoch_accval)
            log.append(loginfo + '\n')
            topaccinfo = 'topacc: ' + str(topacc) + '\n'
            log.append(topaccinfo)
            print(loginfo)
            print(topaccinfo)
        else:
            # --no_val 固定轮数协议：无验证集、无 best-val 选择，最后一轮权重即最终模型
            novalinfo = 'Val skipped (--no_val): epoch {} done, last-epoch weights kept'.format(epoch)
            log.append(novalinfo + '\n')
            print(novalinfo)
        # 每轮用当前模型在测试集上评估一次（仅记录曲线；有 val 时模型选择以 val 为准，--no_val 时直接用最后一轮）
        testacc = get_features_test(testset)
        epoctestinfo = 'Test@Epoch:{} Acc: {:.4f}'.format(epoch, testacc)
        log.append(epoctestinfo + '\n')
        print(epoctestinfo)
    if valdataset is not None:
        model.load_state_dict(best_model)
        fc.load_state_dict(best_fc)
    torch.save(model.state_dict(), './savemodel/model' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    torch.save(fc.state_dict(), './savemodel/fc' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    get_features_test(testset)


def f1_from_matrix(testResult):
    """从混淆矩阵计算逐类 F1 与 macro-F1（precision 走列方向，recall 走行方向）"""
    k = len(testResult)
    f1s = []
    for c in range(k):
        tp = testResult[c][c]
        col = sum(testResult[i][c] for i in range(k))
        row = sum(testResult[c])
        prec = tp / col if col else 0.0
        rec = tp / row if row else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return f1s, sum(f1s) / k


def get_features_test(dataset, prefix='Test'):
    running_loss = 0.0
    running_corrects = 0
    testResult = [[0 for col in range(len(classtype))] for row in range(len(classtype))]  #
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                features = encode_features(images)
                outputs = fc(features)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            predTemp = preds.cpu().numpy().tolist()  #
            testTemp = labels.cpu().numpy().tolist()  #
            for index in range(len(testTemp)):  #
                testResult[testTemp[index]][predTemp[index]] += 1  #
            running_loss += loss.item() * features.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataset)
        epoch_acc = running_corrects.double() / len(dataset)
        f1s, macro_f1 = f1_from_matrix(testResult)
        print('Answers in testlist:', testResult)  #
        loginfo = prefix + ' Loss: {:.4f} Acc: {:.4f} macroF1: {:.4f}'.format(epoch_loss, epoch_acc, macro_f1)
        log.append(loginfo + '\n')
        log.append('PerClassF1: ' + str([round(f, 4) for f in f1s]) + '\n')   # 顺序同 classtype
        log.append(str(testResult))  #
        print(loginfo)

        return round(float(epoch_acc.cpu().detach().numpy()), 6)


class FCN(nn.Module):

    def __init__(self, dim=512, linear_head=False):
        super(FCN, self).__init__()
        if linear_head:
            # WiSE-FT --wise_init 用：单层线性头（论文的零样本分类器本就是线性的）
            self.classifier = nn.Linear(dim, len(classtype))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(dim, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, len(classtype)),
            )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class VPTVisual(nn.Module):
    """VPT（Visual Prompt Tuning, Jia et al. ECCV 2022）Shallow 变体：
    在冻结 CLIP ViT 的 patch 序列中插入 L 个可学习 prompt token。
    序列布局 [CLS, P_1..P_L, patch_1..patch_N]：P 拥有独立可学习位置编码 prompt_pos，
    CLS/patch 沿用原 positional_embedding。仅 prompt/prompt_pos 可训练（约 L×768×2 参数），
    梯度经冻结注意力层回流至 P——前向严禁包 no_grad（optimizer_fc 同时更新 fc 头与 P）。
    与 MASP（AAAI-26）软提示同思想的视觉侧落地：MASP 把可学习 token 拼在 LLM 输入前，
    VPT 拼在 ViT patch 序列前。注意 prompt 参数保持 fp32（Adam 数值稳定），前向时 .to(x.dtype)
    对齐 fp16 主干；本类必须在 clip.model.convert_weights(model) 之后包装。"""

    def __init__(self, visual, vpt_l=32):
        super().__init__()
        self.vt = visual
        self.vpt_l = vpt_l
        width = visual.transformer.width
        self.prompt = nn.Parameter(torch.randn(vpt_l, width) * 0.02)
        self.prompt_pos = nn.Parameter(torch.zeros(vpt_l, width))
        for p in self.vt.parameters():
            p.requires_grad_(False)

    @property
    def input_resolution(self):
        return self.vt.input_resolution          # --aug 与 clip._transform 需要

    @property
    def conv1(self):
        # CLIP.dtype property 内部访问 self.visual.conv1.weight.dtype——包装后必须透传，
        # 否则 encode_image 的 image.type(self.dtype) 会报 "'CLIP' object has no attribute 'dtype'"
        return self.vt.conv1

    def forward(self, x):
        vt = self.vt
        x = vt.conv1(x)                                               # [B, d, g, g]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)    # [B, N, d]
        cls = vt.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)                                # [B, 1+N, d]
        P = self.prompt.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)   # 内容=prompt，位置=prompt_pos
        x = torch.cat([x[:, :1, :], P, x[:, 1:, :]], dim=1)           # [B, 1+L+N, d]
        pos = torch.cat([vt.positional_embedding[:1].to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1),
                         self.prompt_pos.to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1),
                         vt.positional_embedding[1:].to(x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)],
                        dim=1)  # 对齐 [CLS, P, patch]，各段补 batch 维后沿序列维拼接
        x = vt.ln_pre(x + pos)
        x = x.permute(1, 0, 2)                                        # NLD -> LND
        x = vt.transformer(x)
        x = x.permute(1, 0, 2)                                        # LND -> NLD
        x = vt.ln_post(x[:, 0, :])
        return x @ vt.proj


# WiSE-FT --wise_init 的零样本 prompt 模板（每类对全部模板取平均嵌入；可按需增删）
ZS_TEMPLATES = [
    'a photo of {}.',
    'a blurry photo of {}.',
    'a low resolution photo of {}.',
    'a cropped photo of {}.',
    'a bright photo of {}.',
    'a dark photo of {}.',
    'a photo of a person expressing {}.',
    'a photo of a scene evoking {}.',
]


def build_zero_shot_head(model, device, fcdim):
    """WiSE-FT（Wortsman et al. 2022）附录 D.2 配方：用零样本分类器初始化最终分类层。
    每类对 ZS_TEMPLATES 做文本嵌入平均 → L2 归一化 → 乘预训练温度参数（logit_scale≈100），
    微调期间不再含温度参数（偏置清零）。头为单层线性（linear_head=True）——论文的零样本
    分类器本就是线性的，与 MLP 臂存在架构差异，归因时注意。配套：encode_features 会在
    --wise_init 时对图像特征 L2 归一化（论文要求零样本分类器接收归一化的编码器输出）。"""
    with torch.no_grad():
        rows = []
        for c in classtype:
            toks = clip.tokenize([t.format(c) for t in ZS_TEMPLATES]).to(device)
            e = model.encode_text(toks).float()                     # [T, d]，fp16 模型输出转 fp32
            e = e / e.norm(dim=-1, keepdim=True)
            rows.append(e.mean(dim=0))
        W = torch.stack(rows)                                       # [K, d]
        W = W / W.norm(dim=-1, keepdim=True)
        W = model.logit_scale.exp().float() * W                     # ×温度参数
        head = FCN(dim=fcdim, linear_head=True).to(device)
        head.classifier.weight.data.copy_(W)
        head.classifier.bias.data.zero_()
    return head


def encode_features(images):
    """统一的特征编码入口：--wise_init 时对编码器输出做 L2 归一化（论文 D.2 要求），
    其余臂与原行为逐字节一致。getattr 保证 test_saved_model.py 注入的精简 args 不报错"""
    feats = model.encode_image(images).float()
    if getattr(args, 'wise_init', False):
        feats = F.normalize(feats, dim=-1)
    return feats


def _to_rgb(image):
    return image.convert('RGB')


# 说明：--aug 训练增强为 wise-ft 官方实现（其修改版 clip/clip.py 的 _transform(is_train=True)）：
# RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=BICUBIC) + 转 RGB + ToTensor + CLIP Normalize。
# 无翻转、无颜色抖动，与论文附录 D.2 "randomly cropping a square from resized images" 一致。


class FuzzyAwareLoss(nn.Module):
    """FAL（Zheng et al. 2025, Eq.5）：CE + 有界模糊项 -Σ_{i≠ŷ} p_i log p_i，无超参数"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target)                                   # -log p_ŷ
        p = logp.exp()
        idx = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        mask = (idx != target.unsqueeze(1)).float()                     # i != ŷ
        fuzzy = -(p * logp * mask).sum(1).mean()                        # 模糊项，0 < fuzzy ≤ (K-1)e^-1
        return ce + fuzzy


class ListMLELoss(nn.Module):
    """CE + λ·ListMLE（To Err Like Human, NeurIPS 2024 官方实现，λ=0.2）
    情绪序取自官方 repo 的 8×8 矩阵（self=8 最优，Mikel 轮），需从其类别序重排到本项目 classtype 顺序"""
    ECC_ORDER = ['amusement', 'contentment', 'awe', 'excitement', 'fear', 'sadness', 'disgust', 'anger']
    ECC_RANK = [[8, 7, 6, 5, 1, 2, 3, 4],
                [7, 8, 7, 6, 2, 1, 2, 3],
                [6, 7, 8, 7, 3, 2, 1, 2],
                [5, 6, 7, 8, 4, 3, 2, 1],
                [1, 2, 3, 4, 8, 7, 6, 5],
                [2, 1, 2, 3, 7, 8, 7, 6],
                [3, 2, 1, 2, 6, 7, 8, 7],
                [4, 3, 2, 1, 5, 6, 7, 8]]

    def __init__(self, classtype, lam=0.2):
        super().__init__()
        if any(c not in self.ECC_ORDER for c in classtype):
            raise ValueError('ListMLELoss 仅支持 Mikels 8 类情绪数据集（FI_8/EmoSet）')
        perm = [self.ECC_ORDER.index(c) for c in classtype]             # 官方序 -> 本项目序
        k = len(classtype)
        rank = [[self.ECC_RANK[perm[i]][perm[j]] - 1 for j in range(k)] for i in range(k)]
        self.register_buffer('rank', torch.tensor(rank, dtype=torch.float32))
        self.lam = lam

    def listmle_term(self, logits, target):
        """纯 ListMLE 排序项（不含 CE），返回逐样本损失 [B]"""
        true_rank = self.rank[target]
        random_indices = torch.randperm(logits.shape[-1], device=logits.device)  # 官方做法：随机打散处理并列
        pred_shuffled = logits[:, random_indices]
        true_shuffled = true_rank[:, random_indices]
        true_sorted, indices = true_shuffled.sort(descending=True, dim=-1)
        mask = true_sorted == -1
        pred_sorted = torch.gather(pred_shuffled, dim=1, index=indices)
        pred_sorted = pred_sorted.masked_fill(mask, float('-inf'))
        pred_minus_max = pred_sorted - pred_sorted.max(dim=1, keepdim=True).values
        cumsums = torch.cumsum(pred_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        obs_loss = torch.log(cumsums + 1e-7) - pred_minus_max
        obs_loss = obs_loss.masked_fill(mask, 0.0)
        return torch.sum(obs_loss, dim=1)

    def forward(self, logits, target):
        return F.cross_entropy(logits, target) + self.lam * self.listmle_term(logits, target).mean()


class FALListMLELoss(nn.Module):
    """方案A组合：FAL + λ·p_ŷ·ListMLE（座位替换式，FAL 坐进 CE 的座位）
    p_ŷ 置信度调制：伪标签可信（p_ŷ 大）时排序约束全力生效，可疑时自动调低；
    且 p_ŷ·(-log P_s(π)) 在 p_ŷ→0 时趋于 0，保住 FAL 的有界性与噪声容忍性（Theorem 1）"""
    def __init__(self, classtype, num_classes=8, lam=0.2):
        super().__init__()
        self.fal = FuzzyAwareLoss(num_classes=num_classes)
        self.mle = ListMLELoss(classtype, lam=lam)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        conf = logp.exp().gather(1, target.unsqueeze(1)).squeeze(1)     # p_ŷ，与 Focal/FAL 哲学一致，不 detach
        return self.fal(logits, target) + self.mle.lam * (conf * self.mle.listmle_term(logits, target)).mean()


class CEFALMLELoss(nn.Module):
    """无调制三合一：CE + fuzzy项 + λ·ListMLE（三种成分各计一次，排序项不带 p_ŷ 调制）
    与 fal_listmle 唯一区别是去掉 p_ŷ 置信度调制，构成消融对照——隔离调制的贡献"""
    def __init__(self, classtype, num_classes=8, lam=0.2):
        super().__init__()
        self.fal = FuzzyAwareLoss(num_classes=num_classes)
        self.mle = ListMLELoss(classtype, lam=lam)

    def forward(self, logits, target):
        return self.fal(logits, target) + self.mle.lam * self.mle.listmle_term(logits, target).mean()


class DistSoftFALLoss(nn.Module):
    """方案B：情绪距离感知的 FAL——one-hot 标签换成 Mikel 轮距离软标签 q̃_i ∝ exp(-d(i,ŷ)/τ)
    ℓ = -Σ_i [(1-p_i)·q̃_i + p_i]·log(p_i)，d 即 ECC/EMC 的 W-1（自身 0，同极性 1-3，跨极性 4-7）
    软标签真值类恒最大，排序约束不与置信度打架；-p·log p 结构不变，有界性保持
    τ 为温度超参（默认 1.0）：τ 越小软标签越尖，越接近 one-hot"""
    WHEEL = ['anger', 'amusement', 'contentment', 'awe', 'excitement', 'fear', 'sadness', 'disgust']
    POS = {'amusement', 'awe', 'contentment', 'excitement'}

    def __init__(self, classtype, tau=1.0):
        super().__init__()
        if any(c not in self.WHEEL for c in classtype):
            raise ValueError('DistSoftFALLoss 仅支持 Mikels 8 类情绪数据集（FI_8/ArtPhoto/EmoSet），'
                             'EmotionROI 的 joy/surprise 无情绪距离定义')
        # 距离矩阵 d[i][j] = W(真值类 i, 类别 j) - 1，从官方口径（同极性 1+轮距 / 跨极性 4+轮距）构造
        k = len(classtype)
        d = torch.zeros(k, k)
        for i, a in enumerate(classtype):
            for j, b in enumerate(classtype):
                ia, ib = self.WHEEL.index(a), self.WHEEL.index(b)
                dist = min(abs(ia - ib), k - abs(ia - ib))
                d[i, j] = dist if (a in self.POS) == (b in self.POS) else 4 + dist
        soft = F.softmax(-d / tau, dim=1)                     # 行=真值类 y 的软标签 q̃，行和为 1
        self.register_buffer('soft', soft)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        q = self.soft[target]                                 # [B, K] 距离软标签
        focal_soft = -(logp * (1 - p) * q).sum(1)             # -Σ (1-p_i)·q̃_i·log p_i
        entropy = -(p * logp).sum(1)                          # -Σ p_i·log p_i（one-hot 时精确退化为 FAL）
        return (focal_soft + entropy).mean()


if __name__ == '__main__':
    # load settings
    parser = argparse.ArgumentParser(description='Image Emotion Classification by SimEmotion')
    parser.add_argument('--datasetname', type=str, default='FI_8')
    parser.add_argument('--modelname', type=str, default='ViT-L/14', help='仅支持 ViT-B/16（fcdim=512）')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--lr_BB', type=float, default=1e-6)
    parser.add_argument('--wd_BB', type=float, default=1e-5)
    parser.add_argument('--stepsize_BB', type=int, default=6)
    parser.add_argument('--gamma_BB', type=float, default=0.5)
    parser.add_argument('--lr_FC', type=float, default=1e-3)
    parser.add_argument('--wd_FC', type=float, default=0.0,
                        help='FC 头 weight decay；>0 时头优化器切换为 AdamW（WiSE-FT 论文配方），=0 保持 Adam 不变')
    parser.add_argument('--wise_init', action='store_true',
                        help='WiSE-FT 论文配方（附录 D.2）：用零样本文本嵌入×温度初始化单层线性 FC 头，'
                             '并在训练/评估全程 L2 归一化图像特征；测试脚本需加 --wise_init')
    parser.add_argument('--aug', action='store_true',
                        help='WiSE-FT 论文训练增强（官方实现：RandomResizedCrop scale=0.9-1.0 BICUBIC）；'
                             '仅作用于训练集，val/test 仍为确定性 eval preprocess；默认关闭')
    parser.add_argument('--stepsize_FC', type=int, default=6)
    parser.add_argument('--gamma_FC', type=float, default=0.5)
    parser.add_argument('--warmepoch', type=int, default=3)
    parser.add_argument('--datadir', type=str, default=None, help='覆盖 FI_8 数据目录（如 FI_seed43），结尾斜杠可省略')
    parser.add_argument('--no_val', action='store_true',
                        help='小数据集 k 折协议：不加载 val、不做 best-val 选择，训满 num_epochs 用最后一轮权重做 FinalTest'
                             '（要求 datadir 下无 val/ 也不报错；逐轮 Test 日志仍记录，仅供画曲线，不用于选模型）')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'fal', 'listmle', 'fal_listmle', 'ce_fal_mle', 'dist_fal', 'wce', 'vpt'], help='监督损失：ce=交叉熵，fal=fuzzy-aware loss（FAL），listmle=CE+λ·ListMLE情绪序约束，fal_listmle=FAL+λ·p_ŷ·ListMLE组合，ce_fal_mle=CE+模糊项+ListMLE无调制三合一，dist_fal=方案B距离软标签FAL（τ=1.0），wce=类频率倒数加权CE（PaCo哨兵臂），vpt=VPT视觉提示臂（CE损失+冻结ViT+可学习prompt token，Jia et al. 2022，见 --vpt_l；测试脚本需加 --vpt）')
    parser.add_argument('--vpt_l', type=int, default=32,
                        help='VPT 臂的 prompt token 数 L（仅 --loss vpt 生效，默认 32 与 MASP 软提示同量级）；测试脚本需加 --vpt --vpt_l <L>')
    args = parser.parse_args()
    if args.datadir and not args.datadir.endswith('/'):
        args.datadir += '/'   # 路径按字符串拼接，结尾斜杠必须补齐
    classtype, data_dir = datasetInit(args.datasetname)
    set_seed(args.seed)
    log = []

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.modelname, device)
    clip.model.convert_weights(model)
    fcdim = 512
    if args.wise_init:
        fc = build_zero_shot_head(model, device, fcdim)   # WiSE-FT：零样本分类器初始化单层线性头
    else:
        fc = FCN(dim=fcdim).to(device)
    args.modelname = args.modelname.replace('/', '-')

    # VPT 臂：冻结 ViT，patch 序列中插入可学习 prompt token（Jia et al. 2022 Shallow 变体）
    # 必须在 clip.model.convert_weights(model) 之后包装：主干保持 fp16，prompt 参数保持 fp32
    if args.loss == 'vpt':
        model.visual = VPTVisual(model.visual, vpt_l=args.vpt_l).to(device)

    # --aug 训练增强管线（wise-ft 官方实现：RandomResizedCrop，见上方说明块）
    train_preprocess = None
    if args.aug:
        n_px = model.visual.input_resolution                     # ViT-B/16 = 224
        train_preprocess = Compose([
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),       # CLIP 官方均值/方差，同 eval preprocess
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    # load the dataset
    trainset = Loaddata('train')
    if args.datasetname in ('Twitter1', 'Twitter2', 'Abstract7', 'ArtPhoto5fold'):
        # 官方协议（SimEmotion code L341-344 对非 FI 数据集一律 val=test）：checkpoint 按 test 选择
        # → FinalTest 恒为 test 峰值口径，与论文直接可比；官方无末轮协议，强制禁用 --no_val 以防误用
        # Abstract7（官方 abstract_partition 5-fold）与 ArtPhoto5fold（官方 artphoto_partition 5-fold）
        # 同样按此口径（与 Twitter1/Twitter2 同款）
        args.no_val = False
        valset = Loaddata('test')
    else:
        valset = None if args.no_val else Loaddata('val')   # --no_val：小数据集 k 折协议，不加载 val
    testset = Loaddata('test')

    # load the training setting
    # 监督损失：默认交叉熵；--loss fal 时切换为 FAL（train/val/test 共用同一 criterion）
    if args.loss == 'fal':
        criterion = FuzzyAwareLoss(num_classes=len(classtype))
    elif args.loss == 'listmle':
        criterion = ListMLELoss(classtype).to(device)
    elif args.loss == 'fal_listmle':
        criterion = FALListMLELoss(classtype, num_classes=len(classtype)).to(device)
    elif args.loss == 'ce_fal_mle':
        criterion = CEFALMLELoss(classtype, num_classes=len(classtype)).to(device)
    elif args.loss == 'dist_fal':
        criterion = DistSoftFALLoss(classtype).to(device)
    elif args.loss == 'wce':
        # 类频率倒数加权 CE（sklearn balanced 口径）：w_c = N / (K·n_c)，均值 1；少数类梯度话语权提升
        counts = [sum(1 for _, l in trainset.images if l == c) for c in classtype]
        N = sum(counts)
        w = torch.tensor([N / (len(classtype) * n) if n else 0.0 for n in counts], dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=w).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=args.wd_BB)
    # VPT 臂：P token / prompt_pos 并入 optimizer_fc 与 fc 头同组更新（backbone 冻结，optimizer 永不步进）
    head_params = list(fc.parameters())
    if args.loss == 'vpt':
        head_params += [p for p in model.visual.parameters() if p.requires_grad]
    if args.wd_FC > 0:
        optimizer_fc = optim.AdamW(head_params, lr=args.lr_FC, weight_decay=args.wd_FC)  # WiSE-FT 配方
    else:
        optimizer_fc = optim.Adam(head_params, lr=args.lr_FC)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize_BB, gamma=args.gamma_BB)
    scheduler_fc = lr_scheduler.StepLR(optimizer_fc, step_size=args.stepsize_FC, gamma=args.gamma_FC)

    # main phase
    print("New experiment start.")
    starttime = time.strftime('%Y%m%d%H%M%S')
    print(starttime)
    startpoint = time.time()
    get_features(trainset, valset)
    acc = 100 * get_features_test(testset)
    endpoint = time.time()
    timecost = endpoint - startpoint
    print('cost: ' + str(timecost) + 's.')

    # log saving
    paraminfo = args.modelname + '_' +args.datasetname + ' loss:' + args.loss + ' lr_FC:' + str(args.lr_FC) + ' stepsize_FC:' \
                + str(args.stepsize_FC) + ' gamma_FC' + str(args.gamma_FC) + ' lr_BB:' + str(args.lr_BB) \
                + ' stepsize_BB:' + str(args.stepsize_BB) + ' gamma_BB' + str(args.gamma_BB) \
                + ' seed:' + str(args.seed) + ' wd_FC:' + str(args.wd_FC) \
                + (' wise_init:True' if args.wise_init else '') \
                + (' aug:True' if args.aug else '') \
                + (' vpt_l:' + str(args.vpt_l) if args.loss == 'vpt' else '') \
                + (' no_val:True' if args.no_val else '') \
                + (' datadir:' + args.datadir if args.datadir else '')
    print(paraminfo)
    log.append(paraminfo + '\n')
    with open('./log/' + args.modelname + '_' + args.datasetname + str(acc) + '_' + str(starttime) + '.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')
