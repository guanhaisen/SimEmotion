"""prepare_emoset.py — 按官方 80/5/15 划分生成 emoset_split/{train,val,test}/<小写类名>/

依据 emoset_split_json/{train,val,test}.json（官方 EmoSet-118K 划分，2026-11-10 经 Range 方案从
Dropbox 数据包精确拉取并 100% 校验——零缺失零标签冲突，条目格式 [类名, image相对路径, annotation路径]）。
源数据 D:/Coding/Python_Projects/SimEmotion/dataset/Emoset/<类名>/<类名>_<id>.jpg（118,102 张全集）。
类名目录已小写（8 类本身全小写）；幂等可重跑（shutil.copy2 覆盖同名）。
注意：118k 张复制约 15GB 磁盘 / 10-20 分钟；训练用量 train 94,481 ≈ FI 的 5 倍（25ep ~10h/臂）。
"""
import json, os, shutil

SPLIT = r'D:\Coding\Python_Projects\SimEmotion\emoset_split_json'
SRC = r'D:\Coding\Python_Projects\SimEmotion\dataset\Emoset'
OUT = r'D:\Coding\Python_Projects\SimEmotion\emoset_split'

CLS = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

# 源图索引：类名 -> {文件名: 全路径}
src_idx = {}
for c in CLS:
    d = os.path.join(SRC, c)
    src_idx[c] = {fn: os.path.join(d, fn) for fn in os.listdir(d)}
total_src = sum(len(v) for v in src_idx.values())
print(f'源数据: {total_src} 张')

total = 0
seen = set()
for split in ('train', 'val', 'test'):
    items = json.load(open(os.path.join(SPLIT, f'{split}.json'), encoding='utf-8'))
    counts = {c: 0 for c in CLS}
    for it in items:
        cls, rel = it[0], it[1]
        fn = os.path.basename(rel)
        assert cls in src_idx and fn in src_idx[cls], f'{split}: {cls}/{fn} 不在源数据'
        dst_dir = os.path.join(OUT, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src_idx[cls][fn], os.path.join(dst_dir, fn))
        counts[cls] += 1
        total += 1
        seen.add(fn)
    print(f'{split}: {counts} 合计 {len(items)}')
print(f'落盘总数: {total}（唯一文件 {len(seen)}）')

assert total == len(seen) == total_src == 118102, '落盘总数/唯一数/源数不一致'
print('校验通过：118,102 张全部落盘，三划分互斥（唯一文件数 = 总数）')
