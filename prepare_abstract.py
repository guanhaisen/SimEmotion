# -*- coding: utf-8 -*-
"""一次性准备脚本：Abstract（280 张抽象画，imageemotion.org，Machajdik & Hanbury）单标签分层划分

标签规则：ABSTRACT_groundTruth.csv 的 8 类情绪投票 argmax 单标签；
  51 张 argmax 并列（两类票数相同）按 seed43 随机分配到并列类之一（分配清单写入 manifest）。
划分：分层 60/20/20 seed43（与 ArtPhoto 同协议）→ abstract_split/{train,val,test}/<类>/（复制不移动）。
局限（写实验记录必须注明）：
  1. test 仅 56 张 → 单 split 噪声带 ±5-6pp（ArtPhoto 162 张为 ±2-3pp），结论定性为主，需同时报 test peak；
  2. anger 全集仅 9 张 → test 约 2 张，per-class anger 指标无意义；
  3. 51 张并列分配是已知标签噪声（这批图本就是情绪最模糊的图）。
运行前先删除已存在的 abstract_split/。
"""
import csv
import os
import random
import shutil
from collections import Counter

IMG_DIR = r'D:\Coding\Python_Projects\SimEmotion\abstract'
CSV_PATH = os.path.join(IMG_DIR, 'ABSTRACT_groundTruth.csv')
DST = r'D:\Coding\Python_Projects\SimEmotion\abstract_split'
CLASSES = ['amusement', 'anger', 'awe', 'contentment', 'disgust',
           'excitement', 'fear', 'sadness']
# CSV 列名是缩写（Content/Sad），与项目类名的显式映射
HEADER = {'amusement': 'amusement', 'anger': 'anger', 'awe': 'awe',
          'contentment': 'content', 'disgust': 'disgust',
          'excitement': 'excitement', 'fear': 'fear', 'sadness': 'sad'}
SEED = 43
RATIO = (0.6, 0.2, 0.2)   # train/val/test，与 ArtPhoto 同协议


def load_labels():
    """读 CSV → {fname: (winners列表, 票数最大值)}；winners 为 argmax 并列类列表"""
    with open(CSV_PATH, encoding='utf-8-sig', errors='ignore') as f:
        rows = list(csv.reader(f))
    header = [h.strip().strip("'\"").lower() for h in rows[0]]
    emo_cols = [header.index(HEADER[c]) for c in CLASSES]
    labels = {}
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        fname = row[0].strip().strip("'\"")
        votes = [int(row[i]) for i in emo_cols]
        m = max(votes)
        if m <= 0:
            raise SystemExit(f'{fname} 全零票（数据异常）')
        winners = [CLASSES[k] for k, v in enumerate(votes) if v == m]
        labels[fname] = (winners, m)
    return labels


def stratified_split(names, rng):
    """类内 shuffle 后按 60/20/20 切分（val/test 用 round，至少给 val/test 各留 1 张若类大小>=3）"""
    rng.shuffle(names)
    n = len(names)
    n_val = max(1, round(n * RATIO[1])) if n >= 3 else (1 if n == 2 else 0)
    n_test = max(1, round(n * RATIO[2])) if n >= 3 else (1 if n == 2 else 0)
    n_val = min(n_val, n - 1) if n >= 2 else 0
    n_test = min(n_test, n - n_val) if n >= 2 else 0
    val = names[:n_val]
    test = names[n_val:n_val + n_test]
    train = names[n_val + n_test:]
    return train, val, test


def main():
    labels = load_labels()
    rng = random.Random(SEED)

    if os.path.exists(DST):
        raise SystemExit(f'{DST} 已存在，请先删除再运行（防止旧 split 混入）')

    # 1) argmax + 并列分配（seed43 固定）
    assign = {}          # {fname: 类}
    tie_log = []         # 并列分配清单（进 manifest）
    for fname, (winners, m) in labels.items():
        if len(winners) > 1:
            cls = winners[rng.randrange(len(winners))]
            tie_log.append((fname, '/'.join(winners), cls))
        else:
            cls = winners[0]
        assign[fname] = cls

    # 2) 类内分层 60/20/20
    by_class = {c: [] for c in CLASSES}
    for fname, cls in assign.items():
        by_class[cls].append(fname)
    splits = {'train': [], 'val': [], 'test': []}
    dist = {s: Counter() for s in splits}
    for c in CLASSES:
        tr, va, te = stratified_split(by_class[c], rng)
        for s, items in (('train', tr), ('val', va), ('test', te)):
            for f in items:
                dst_dir = os.path.join(DST, s, c)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(os.path.join(IMG_DIR, f), os.path.join(dst_dir, f))
                dist[s][c] += 1
            splits[s].extend((c, f) for f in items)

    # 3) 覆盖检查
    assert len(splits['train']) + len(splits['val']) + len(splits['test']) == len(assign)
    all_dst = {f for s in splits for _, f in splits[s]}
    assert len(all_dst) == len(assign), '存在重复分配'

    # 4) 汇总
    print(f'Abstract 划分完成（seed{SEED}，分层 60/20/20）：')
    for s in ('train', 'val', 'test'):
        row = ' '.join(f'{c}={dist[s][c]}' for c in CLASSES)
        print(f'  {s}: {len(splits[s])} 张 | {row}')
    print(f'\n并列图分配：{len(tie_log)} 张（seed{SEED} 随机分到并列类之一）')

    # 5) manifest
    with open(os.path.join(os.path.dirname(DST), 'abstract_split_manifest.txt'),
              'w', encoding='utf-8') as f:
        f.write(f'# Abstract split seed{SEED} stratified 60/20/20（与 ArtPhoto 同协议）\n')
        f.write('# 标签规则: CSV 8 类票数 argmax；并列图 seed43 随机分配（下列清单）\n')
        f.write('# 局限: test 56 张噪声带 ±5-6pp；anger test≈2 张 per-class 无意义\n')
        f.write(f"# train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}\n")
        for s in ('train', 'val', 'test'):
            for c, f_ in sorted(splits[s]):
                f.write(f'{s}\t{c}\t{f_}\n')
        f.write('\n# 并列图分配清单（fname\t并列类\tr分配结果）\n')
        for fname, w, cls in tie_log:
            f.write(f'TIE\t{fname}\t{w}\t{cls}\n')
    print('manifest: abstract_split_manifest.txt（含并列分配清单）')


if __name__ == '__main__':
    main()
