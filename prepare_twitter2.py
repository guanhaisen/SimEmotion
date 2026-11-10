"""prepare_twitter2.py — 按官方 5-fold 生成 twitter2_fold{i}/{train,test}/{pos,neg}/

依据 d:/VibeCoding/Trae/project/task/config/dataset_split/twitter603/：
  train_{i}.txt（"文件名 标签"格式，0=neg 1=pos，已与 groundTruthLabel.txt 100% 校验一致）
  partition{i}.txt（test 列表，仅文件名，标签取 groundTruthLabel.txt）
源数据 D:/Coding/Python_Projects/SimEmotion/dataset/Twitter2/{pos,neg}（470/133 = 603 张）。
注意：官方每折仅覆盖 587/603 张（16 张不进任何折，官方划分本身如此），幂等可重跑。
"""
import os, shutil

SRC = r'D:\Coding\Python_Projects\SimEmotion\dataset\Twitter2'
CFG = r'D:\VibeCoding\Trae\project\task\config\dataset_split\twitter603'
OUT = r'D:\Coding\Python_Projects\SimEmotion'

# ground truth（Image<TAB>Sentiment）：partition{i}.txt 无标签，从这取
gt = {}
with open(os.path.join(SRC, 'groundTruthLabel.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.split()
        if len(parts) >= 2:
            gt[parts[0]] = parts[1]   # 'positive' / 'negative'

img2src = {}
for c in ['pos', 'neg']:
    for f in os.listdir(os.path.join(SRC, c)):
        img2src[f] = os.path.join(SRC, c, f)

def cls_of(fn):
    """先查 GT（权威），缺失时回退目录归属"""
    if fn in gt:
        return 'pos' if gt[fn] == 'positive' else 'neg'
    for c in ('pos', 'neg'):
        if os.path.exists(os.path.join(SRC, c, fn)):
            return c
    raise AssertionError(f'{fn} 不在源数据也不在 groundTruth')

total = 0
for i in range(1, 6):
    fold_counts = {'pos': 0, 'neg': 0}
    for split, fname, labeled in [('train', f'train_{i}.txt', True), ('test', f'partition{i}.txt', False)]:
        for line in open(os.path.join(CFG, fname), encoding='utf-8', errors='ignore'):
            if not line.strip():
                continue
            parts = line.split()
            fn = parts[0]
            if labeled:
                cls = 'pos' if int(parts[1]) == 1 else 'neg'
            else:
                cls = cls_of(fn)
            dst_dir = os.path.join(OUT, f'twitter2_fold{i}', split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            src = img2src.get(fn)
            assert src is not None, f'missing {fn}'
            shutil.copy2(src, os.path.join(dst_dir, fn))
            fold_counts[cls] += 1
            total += 1
    print(f'fold{i}:', fold_counts, 'train+test', sum(fold_counts.values()))
print('total copied:', total, '(= 5 × 587；官方每折不覆盖 16 张)')
