# -*- coding: utf-8 -*-
"""一次性准备脚本：Abstract 无 val 划分（train/test 二分）
规则：test 56 张原样保留（与 60/20/20 split 的 test 同集，结果可直接对比），
  原 val 56 张并入 train → train 224 / test 56。
配合 main.py --no_val 固定轮数协议使用（无 best-val 选择，最后一轮权重即最终模型）。
运行前先删除已存在的 abstract_split_noval/。
"""
import os
import shutil

SRC = r'D:\Coding\Python_Projects\SimEmotion\abstract_split'
DST = r'D:\Coding\Python_Projects\SimEmotion\abstract_split_noval'
SPLITS = ['train', 'val', 'test']

if os.path.exists(DST):
    raise SystemExit(f'{DST} 已存在，请先删除再运行')

n_copied = 0
for cls in sorted(os.listdir(os.path.join(SRC, 'test'))):
    for split, dst_split in (('train', 'train'), ('val', 'train'), ('test', 'test')):
        src_dir = os.path.join(SRC, split, cls)
        dst_dir = os.path.join(DST, dst_split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for f in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            n_copied += 1

# 校验
total = 0
for split in ('train', 'test'):
    per = {c: len(os.listdir(os.path.join(DST, split, c))) for c in sorted(os.listdir(os.path.join(DST, split)))}
    n = sum(per.values())
    total += n
    print(f'{split}: {n} | {per}')
assert total == 280, total
print(f'完成：共 {n_copied} 张复制到 {DST}（train=原train+原val, test=原test 同集）')
