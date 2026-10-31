# -*- coding: utf-8 -*-
"""一次性准备脚本：按 Trae task 的 abstract_partition 5 折清单重建 Abstract 划分。

划分来源：d:\\VibeCoding\\Trae\\project\\task\\config\\dataset_split\\abstract_partition
  train{1..5}.txt / test{1..5}.txt，每行 "abstract_XXXX.jpg <label_id>"，每折覆盖 226 张子集
数据源：abstract\\（280 张完整版平铺；划分只覆盖其中 226 张，其余跳过）
label_id→类别映射：从 artphoto_partition（文件名自带类别）反推，并与 ABSTRACT_groundTruth.csv
  的 8 类票数 argmax 做一致性抽查后采用。
输出：abstract_5fold\\fold{i}\\{train,test}\\<类>\\（按清单复制，不移动）
运行前先删除已存在的 abstract_5fold/。
"""
import os
import shutil
import csv
from collections import defaultdict

SPLIT_DIR = r'd:\VibeCoding\Trae\project\task\config\dataset_split'
IMG_DIR = r'D:\Coding\Python_Projects\SimEmotion\abstract'
CSV_PATH = os.path.join(IMG_DIR, 'ABSTRACT_groundTruth.csv')
DST = r'D:\Coding\Python_Projects\SimEmotion\abstract_5fold'
CLASSES = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
           'excitement', 'fear', 'sad']


def derive_mapping_from_artphoto():
    """artphoto 文件名自带类别，从其 5 折清单直接反推 label_id → 类别（应每类唯一）"""
    mapping = {}
    conflicts = defaultdict(set)
    for i in range(1, 6):
        for name in ('train', 'test'):
            path = os.path.join(SPLIT_DIR, 'artphoto_partition', f'{name}{i}.txt')
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    cls = parts[0].rsplit('_', 1)[0]
                    lb = parts[1]
                    if cls in mapping and mapping[cls] != lb:
                        conflicts[cls].add(mapping[cls])
                    conflicts[cls].add(lb)
                    mapping[cls] = lb
    bad = {c: v for c, v in conflicts.items() if len(v) > 1}
    if bad:
        raise SystemExit(f'artphoto label 映射有冲突: {bad}')
    # 统一转为 int → 类别
    m = {int(v): k for k, v in mapping.items()}
    if sorted(m) != list(range(8)):
        raise SystemExit(f'artphoto label 映射不完整: {sorted(m)}')
    return m


def load_csv_argmax():
    """读 groundTruth CSV，返回 {fname: argmax类别}（票数最高者，仅作抽查对照）"""
    with open(CSV_PATH, 'r', encoding='utf-8-sig', errors='ignore') as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip("'\"") for h in rows[0]]
    # 找出 8 个情感列
    emo_cols = [i for i, h in enumerate(header) if h.lower() in CLASSES]
    argmax = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        fname = row[0].strip().strip("'\"")
        try:
            votes = [int(row[i]) for i in emo_cols]
        except ValueError:
            continue
        argmax[fname] = CLASSES[votes.index(max(votes))]
    return argmax


def main():
    mapping = derive_mapping_from_artphoto()
    print('label_id → 类别 映射（源自 artphoto 划分）:')
    for k in sorted(mapping):
        print(f'  {k} = {mapping[k]}')

    csv_argmax = load_csv_argmax()

    if os.path.exists(DST):
        raise SystemExit(f'{DST} 已存在，请先删除再运行（防止重复/混入）')

    all_src = {f for f in os.listdir(IMG_DIR) if f.lower().endswith('.jpg')}
    covered = set()
    agree = agree_n = 0
    for i in range(1, 6):
        fold = {}
        for name in ('train', 'test'):
            path = os.path.join(SPLIT_DIR, 'abstract_partition', f'{name}{i}.txt')
            items = []
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for lineno, line in enumerate(f, 1):
                    parts = line.split()
                    if not parts:
                        continue
                    fname, lb = parts[0], int(parts[1])
                    if lb not in mapping:
                        raise SystemExit(f'fold{i} {name} 第 {lineno} 行 label 非法: {lb}')
                    cls = mapping[lb]
                    if fname not in all_src:
                        raise SystemExit(f'fold{i} 源文件缺失: {fname}')
                    # 与 CSV argmax 抽查一致性
                    if fname in csv_argmax:
                        agree_n += 1
                        if csv_argmax[fname] == cls:
                            agree += 1
                    items.append((cls, fname))
            fold[name] = items
        keys = [(c, f) for name in ('train', 'test') for c, f in fold[name]]
        if len(keys) != len(set(keys)):
            raise SystemExit(f'fold{i} 存在重复条目')
        for name in ('train', 'test'):
            for cls, fname in fold[name]:
                dst_dir = os.path.join(DST, f'fold{i}', name, cls)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(os.path.join(IMG_DIR, fname), os.path.join(dst_dir, fname))
                covered.add(fname)
        dist = {c: sum(1 for cc, _ in fold['test'] if cc == c) for c in CLASSES}
        print(f'fold{i}: train={len(fold["train"])} test={len(fold["test"])}（test 分布 {dist}）')

    missing = sorted(all_src - covered)
    print(f'划分覆盖 {len(covered)} 张；未进入划分的源图 {len(missing)} 张（跳过）')

    if agree_n:
        print(f'与 CSV argmax 标注一致性抽查：{agree}/{agree_n} = {agree / agree_n:.1%}'
              f'（若偏低说明划分标签用了其他规则，映射仍以 artphoto 为准）')

    with open(os.path.join(DST, 'split_manifest.txt'), 'w', encoding='utf-8') as f:
        f.write('# Abstract 5-fold split（来源：task/config/dataset_split/abstract_partition，无 val）\n')
        f.write('# label 映射: ' + ', '.join(f'{k}={mapping[k]}' for k in sorted(mapping)) + '\n')
        for i in range(1, 6):
            for name in ('train', 'test'):
                path = os.path.join(SPLIT_DIR, 'abstract_partition', f'{name}{i}.txt')
                with open(path, 'r', encoding='utf-8', errors='ignore') as g:
                    for line in g:
                        line = line.strip()
                        if line:
                            fname, lb = line.split()
                            f.write(f'fold{i}\t{name}\t{fname}\t{mapping[int(lb)]}\n')

    print(f'完成：5 折共复制 {5 * 226} 张到 {DST}')


if __name__ == '__main__':
    main()
