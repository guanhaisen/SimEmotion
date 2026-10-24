# -*- coding: utf-8 -*-
"""一次性准备脚本：把扁平目录的 ArtPhoto（806 张，文件名前缀=类别）整理成 FI 同构的
artphoto_split/{train,val,test}/<class>/ 结构。

划分：分层随机 60/20/20，seed 43（ArtPhoto 仅 806 张，5% val 每类只有 3-5 张，checkpoint 选择不稳）
类别映射：sad → sadness（对齐 FI_8 的 Mikels 8 类 taxonomy）
数据复制不移动，原目录保持原样；划分清单写入 split_manifest.txt 供复现与查重。
运行一次即可，重复运行前先删除 artphoto_split/。
"""
import os
import random
import shutil

SEED = 43
SRC = r'D:\Coding\Python_Projects\SimEmotion\artphoto'
DST = r'D:\Coding\Python_Projects\SimEmotion\artphoto_split'
RATIOS = {'train': 0.6, 'val': 0.2, 'test': 0.2}
NAME_MAP = {'sad': 'sadness'}          # 数据集前缀 → Mikels 8 类标准名

def main():
    random.seed(SEED)

    # 1) 扫描并按类别分组
    by_class = {}
    for f in os.listdir(SRC):
        if not f.lower().endswith('.jpg'):
            continue
        prefix = f.rsplit('_', 1)[0]
        cls = NAME_MAP.get(prefix, prefix)
        by_class.setdefault(cls, []).append(f)

    # 2) 分层划分
    split = {'train': [], 'val': [], 'test': []}   # (cls, filename)
    print('逐类划分（60/20/20，seed 43）：')
    for cls in sorted(by_class):
        files = sorted(by_class[cls])
        random.shuffle(files)
        n = len(files)
        n_train = round(n * RATIOS['train'])
        n_val = round(n * RATIOS['val'])
        split['train'] += [(cls, f) for f in files[:n_train]]
        split['val'] += [(cls, f) for f in files[n_train:n_train + n_val]]
        split['test'] += [(cls, f) for f in files[n_train + n_val:]]
        print(f"  {cls:<12} 总数={n:<4} train={n_train} val={n_val} test={n - n_train - n_val}")

    # 3) 复制到目标结构
    if os.path.exists(DST):
        raise SystemExit(f'{DST} 已存在，请先删除再重新运行（防止重复/混入）')
    for setname, items in split.items():
        for cls, fname in items:
            dst_dir = os.path.join(DST, setname, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(os.path.join(SRC, fname), os.path.join(dst_dir, fname))

    # 4) 划分清单（复现 + 查重）
    with open(os.path.join(DST, 'split_manifest.txt'), 'w', encoding='utf-8') as f:
        f.write(f'# ArtPhoto stratified split 60/20/20, seed={SEED}, sad->sadness\n')
        for setname in ('train', 'val', 'test'):
            for cls, fname in split[setname]:
                f.write(f'{setname}\t{cls}\t{fname}\n')

    total = sum(len(v) for v in split.values())
    print(f'\n完成：共复制 {total} 张到 {DST}')
    print('集合分布：', {k: len(v) for k, v in split.items()})

if __name__ == '__main__':
    main()
