# -*- coding: utf-8 -*-
"""一次性准备脚本：把 Cornell EmotionROI（1,980 张，images/<6类>/330）整理成 FI 同构的
emotionroi_split/{train,val,test}/<class>/ 结构。

划分策略（自动选择并打印）：
  优先解析官方 training_testing_split/{training,testing}.txt（Peng et al. ICIP 2016 的划分）：
    - 若每行能解析出图片路径且经 images/<类>/ 定位到唯一文件 → 采用官方 train/test，
      再从官方 train 内分层抽 10% 作为 val（seed 43）
  解析失败则回退：分层随机 80/10/10，seed 43
数据复制不移动；划分清单写入 split_manifest.txt。运行一次即可，重复运行前先删除 emotionroi_split/。
注意：joy/surprise 不在 Mikels 8 类中，本数据集仅支持 --loss ce/fal。
"""
import os
import random
import shutil

SEED = 43
SRC = r'D:\Coding\Python_Projects\SimEmotion\EmotionROI'
IMG_DIR = os.path.join(SRC, 'images')
SPLIT_DIR = os.path.join(SRC, 'training_testing_split')
DST = r'D:\Coding\Python_Projects\SimEmotion\emotionroi_split'
CLASSES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

def load_official_split():
    """解析官方划分文件；成功返回 {setname: [(cls, fname), ...]}，失败返回 None"""
    result = {}
    for setname, txt in (('train', 'training.txt'), ('test', 'testing.txt')):
        path = os.path.join(SPLIT_DIR, txt)
        if not os.path.exists(path):
            return None
        items = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 在 images/ 下定位该文件，类别 = 其所在子目录名
                cand = [os.path.join(IMG_DIR, c, os.path.basename(line))
                        for c in CLASSES]
                hit = [p for p in cand if os.path.exists(p)]
                if len(hit) != 1:
                    return None                      # 有歧义/找不到 → 放弃官方划分
                cls = os.path.basename(os.path.dirname(hit[0]))
                if cls not in CLASSES:
                    return None
                items.append((cls, os.path.basename(hit[0])))
        result[setname] = items
    return result if all(result.values()) else None

def stratified(items, ratios):
    by_class = {}
    for cls, fname in items:
        by_class.setdefault(cls, []).append(fname)
    out = {k: [] for k in ratios}
    for cls in sorted(by_class):
        files = sorted(by_class[cls])
        random.shuffle(files)
        n = len(files)
        n_train = round(n * ratios['train'])
        n_val = round(n * ratios['val'])
        out['train'] += [(cls, f) for f in files[:n_train]]
        out['val'] += [(cls, f) for f in files[n_train:n_train + n_val]]
        out['test'] += [(cls, f) for f in files[n_train + n_val:]]
    return out

def main():
    random.seed(SEED)
    all_files = {c: os.listdir(os.path.join(IMG_DIR, c)) for c in CLASSES}
    print('源数据逐类计数：', {c: len(v) for c, v in all_files.items()})

    official = load_official_split()
    if official:
        # 官方 train 内分层抽 10% 作 val
        train_val = stratified(official['train'], {'train': 0.9, 'val': 0.1})
        split = {'train': train_val['train'], 'val': train_val['val'], 'test': official['test']}
        print('采用划分：官方 training/testing.txt（val = 官方 train 内分层 10%，seed 43）')
    else:
        flat = [(c, f) for c in CLASSES for f in all_files[c]]
        split = stratified(flat, {'train': 0.8, 'val': 0.1, 'test': 0.1})
        print('采用划分：官方解析失败，回退分层随机 80/10/10（seed 43）')

    if os.path.exists(DST):
        raise SystemExit(f'{DST} 已存在，请先删除再重新运行（防止重复/混入）')
    for setname, items in split.items():
        for cls, fname in items:
            dst_dir = os.path.join(DST, setname, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(os.path.join(IMG_DIR, cls, fname), os.path.join(dst_dir, fname))

    with open(os.path.join(DST, 'split_manifest.txt'), 'w', encoding='utf-8') as f:
        f.write(f'# EmotionROI split, seed={SEED}（官方划分优先，回退分层随机）\n')
        for setname in ('train', 'val', 'test'):
            for cls, fname in split[setname]:
                f.write(f'{setname}\t{cls}\t{fname}\n')

    print('集合分布：', {k: len(v) for k, v in split.items()})
    print(f'完成：共复制 {sum(len(v) for v in split.values())} 张到 {DST}')

if __name__ == '__main__':
    main()
