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
        path = data_dir
        self.classindex = {'amusement': 0, 'anger': 1, 'awe': 2, 'contentment': 3, 'disgust': 4, 'excitement': 5,
                           'fear': 6, 'sadness': 7}

        for ct in classtype:
            filepath = path + setname + '/' + ct + '/'
            file_list = os.listdir(filepath)
            for filename in file_list:
                img_path = filepath + filename
                img_label = ct
                self.images.append((img_path, img_label))

    def __getitem__(self, item):
        image, label = self.images[item]
        img = preprocess(Image.open(image)).to(device)
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
            if epoch >= args.warmepoch:
                optimizer.zero_grad()
            optimizer_fc.zero_grad()
            with torch.set_grad_enabled(True):
                features = model.encode_image(images).float()
                outputs = fc(features)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                if epoch >= args.warmepoch:
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

        if epoch >= args.warmepoch:
            scheduler.step()
        scheduler_fc.step()
        running_lossval = 0.0
        running_correctsval = 0

        for images, labels in tqdm(DataLoader(valdataset, batch_size=args.batch_size, shuffle=False)):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                features = model.encode_image(images).float()
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
        # 每轮用当前模型在测试集上评估一次（仅记录曲线，模型选择仍以 val 为准）
        testacc = get_features_test(testset)
        epoctestinfo = 'Test@Epoch:{} Acc: {:.4f}'.format(epoch, testacc)
        log.append(epoctestinfo + '\n')
        print(epoctestinfo)
    model.load_state_dict(best_model)
    fc.load_state_dict(best_fc)
    torch.save(model.state_dict(), './savemodel/model' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    torch.save(fc.state_dict(), './savemodel/fc' + args.modelname + '_' + args.datasetname + '_' + str(starttime) + '.pth')
    get_features_test(testset)


def get_features_test(dataset):
    running_loss = 0.0
    running_corrects = 0
    testResult = [[0 for col in range(len(classtype))] for row in range(len(classtype))]  #
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)):
            images = images.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                features = model.encode_image(images).float()
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
        print('Answers in testlist:', testResult)  #
        loginfo = 'Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc)
        log.append(loginfo + '\n')
        log.append(str(testResult))  #
        print(loginfo)

        return round(float(epoch_acc.cpu().detach().numpy()), 6)


class FCN(nn.Module):

    def __init__(self, dim=512):
        super(FCN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, len(classtype)),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


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
    parser.add_argument('--stepsize_FC', type=int, default=6)
    parser.add_argument('--gamma_FC', type=float, default=0.5)
    parser.add_argument('--warmepoch', type=int, default=3)
    parser.add_argument('--datadir', type=str, default=None, help='覆盖 FI_8 数据目录（如 FI_seed43），结尾斜杠可省略')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'fal', 'listmle', 'fal_listmle'], help='监督损失：ce=交叉熵，fal=fuzzy-aware loss（FAL），listmle=CE+λ·ListMLE情绪序约束，fal_listmle=FAL+λ·p_ŷ·ListMLE组合')
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
    fc = FCN(dim=fcdim).to(device)
    args.modelname = args.modelname.replace('/', '-')

    # load the dataset
    trainset = Loaddata('train')
    valset = Loaddata('val')
    testset = Loaddata('test')

    # load the training setting
    # 监督损失：默认交叉熵；--loss fal 时切换为 FAL（train/val/test 共用同一 criterion）
    if args.loss == 'fal':
        criterion = FuzzyAwareLoss(num_classes=len(classtype))
    elif args.loss == 'listmle':
        criterion = ListMLELoss(classtype).to(device)
    elif args.loss == 'fal_listmle':
        criterion = FALListMLELoss(classtype, num_classes=len(classtype)).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.visual.parameters(), lr=args.lr_BB, weight_decay=args.wd_BB)
    optimizer_fc = optim.Adam(fc.parameters(), lr=args.lr_FC)
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
                + ' stepsize_BB:' + str(args.stepsize_BB) + ' gamma_BB' + str(args.gamma_BB)
    print(paraminfo)
    log.append(paraminfo + '\n')
    with open('./log/' + args.modelname + '_' + args.datasetname + str(acc) + '_' + str(starttime) + '.txt', 'a+') as flog:
        for line in log:
            flog.write(str(line))
        print('Log saved successfully.')
