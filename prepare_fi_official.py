"""prepare_fi_official.py — 按官方划分生成 fi_official_split/

依据 d:/VibeCoding/Trae/project/task/config/ 的官方划分清单：
  fi_train_img_label.txt (18,156) / fi_val_img_label.txt (1,136) / fi_test_img_label.txt (3,408)
官方标签序（已由文件夹归属 100% 验证）：
  0=Amusement 1=Contentment 2=Awe 3=Excitement 4=Fear 5=Sadness 6=Disgust 7=Anger
输出 fi_official_split/{train,val,test}/<小写类名>/，类别文件夹按本项目 classtype 序（amuse/anger/awe/...）。
"""
import os, shutil

ROOT = r'D:\Coding\Python_Projects\SimEmotion\dataset\FI'
OUT = r'D:\Coding\Python_Projects\SimEmotion\fi_official_split'
CFG = r'D:\VibeCoding\Trae\project\task\config'
CLASSES = ['Amusement', 'Anger', 'Awe', 'Contentment', 'Disgust', 'Excitement', 'Fear', 'Sadness']
LABEL2CLS = {0: 'Amusement', 1: 'Contentment', 2: 'Awe', 3: 'Excitement',
             4: 'Fear', 5: 'Sadness', 6: 'Disgust', 7: 'Anger'}

# 索引图像：文件名 -> 源路径（8 类文件夹；排除 backup/root/split_fewshot）
img2src = {}
for c in CLASSES:
    for f in os.listdir(os.path.join(ROOT, c)):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img2src[f] = os.path.join(ROOT, c, f)

total_copied, missing_all = 0, []
for split in ['train', 'val', 'test']:
    lines = [l.split() for l in open(os.path.join(CFG, f'fi_{split}_img_label.txt'),
                                     encoding='utf-8') if l.strip()]
    counts = {c.lower(): 0 for c in CLASSES}
    for fn, lb in lines:
        cls = LABEL2CLS[int(lb)].lower()
        dst_dir = os.path.join(OUT, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        src = img2src.get(fn)
        if src is None:
            missing_all.append((split, fn))
            continue
        shutil.copy2(src, os.path.join(dst_dir, fn))
        counts[cls] += 1
        total_copied += 1
    print(split, counts, 'sum', sum(counts.values()))

print('copied total:', total_copied)
print('missing (%d):' % len(missing_all), missing_all)
# 校验：train+val+test = trainval 行数
tv = sum(1 for l in open(os.path.join(CFG, 'fi_trainval_img_label.txt'), encoding='utf-8') if l.strip())
print('trainval lines:', tv, '== train+val?', tv == 18156 + 1136)
