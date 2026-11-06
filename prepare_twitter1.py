"""prepare_twitter1.py — 按官方 5-fold 生成 twitter1_fold{i}/{train,test}/{pos,neg}/

依据 d:/VibeCoding/Trae/project/task/config/dataset_split/twitter1269/：
  train_{i}.txt / test_{i}.txt（"文件名 标签"格式，0=neg 1=pos，已 100% 校验）
源数据 D:/Coding/Python_Projects/SimEmotion/dataset/Twitter1/{pos,neg}（769/500 = 1,269 张）。
"""
import os, shutil

SRC = r'D:\Coding\Python_Projects\SimEmotion\dataset\Twitter1'
CFG = r'D:\VibeCoding\Trae\project\task\config\dataset_split\twitter1269'
OUT = r'D:\Coding\Python_Projects\SimEmotion'

img2src = {}
for c in ['pos', 'neg']:
    for f in os.listdir(os.path.join(SRC, c)):
        img2src[f] = os.path.join(SRC, c, f)

LBL2CLS = {0: 'neg', 1: 'pos'}
total = 0
for i in range(1, 6):
    for split, fname in [('train', f'train_{i}.txt'), ('test', f'test_{i}.txt')]:
        lines = [l.split() for l in open(os.path.join(CFG, fname), encoding='utf-8',
                                         errors='ignore') if l.strip()]
        counts = {'pos': 0, 'neg': 0}
        for fn, lb in lines:
            cls = LBL2CLS[int(lb)]
            dst_dir = os.path.join(OUT, f'twitter1_fold{i}', split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            src = img2src.get(fn)
            assert src is not None, f'missing {fn}'
            shutil.copy2(src, os.path.join(dst_dir, fn))
            counts[cls] += 1
            total += 1
    print(f'fold{i}:', counts, 'train+test', sum(counts.values()))
print('total copied:', total, '(= 5 × 1269)')
