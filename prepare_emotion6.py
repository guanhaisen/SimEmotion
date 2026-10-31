# -*- coding: utf-8 -*-
"""一次性准备脚本：按 Trae task 的 Emotion6 划分清单重建 train/test 划分。

划分来源：d:\\VibeCoding\\Trae\\project\\task\\config\\dataset_split\\Emotion6
  trainimg_label.txt 1,386 行 / testimg_label.txt 594 行，每行 "<类>/<N>.jpg <label_id>"
数据源：EmotionROI\\images\\<类>\\（1,980 张，每类 330）
输出：Emotion6_split\\{train,test}\\<类>\\（按清单复制，不移动）
说明：该划分沿用原始论文协议，只有 train/test，无 val。
运行前先删除已存在的 Emotion6_split/。
"""
import os
import shutil

SPLIT_DIR = r'd:\VibeCoding\Trae\project\task\config\dataset_split\Emotion6'
IMG_DIR = r'D:\Coding\Python_Projects\SimEmotion\EmotionROI\images'
DST = r'D:\Coding\Python_Projects\SimEmotion\Emotion6_split'
CLASSES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def load_split(txtname):
    """解析 "<类>/<文件名> <label_id>" 清单，返回 [(cls, fname), ...]"""
    path = os.path.join(SPLIT_DIR, txtname)
    items = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2 or '/' not in parts[0]:
                raise SystemExit(f'{txtname} 第 {lineno} 行格式异常: {line!r}')
            rel = parts[0]
            cls = os.path.dirname(rel)
            fname = os.path.basename(rel)
            if cls not in CLASSES:
                raise SystemExit(f'{txtname} 第 {lineno} 行类别非法: {cls}')
            items.append((cls, fname))
    return items


def main():
    split = {name: load_split(f'{name}img_label.txt') for name in ('train', 'test')}

    # 校验：清单内无重复、文件全部存在
    seen = set()
    for name, items in split.items():
        for cls, fname in items:
            key = (cls, fname)
            if key in seen:
                raise SystemExit(f'重复条目: {cls}/{fname}')
            seen.add(key)
            if not os.path.exists(os.path.join(IMG_DIR, cls, fname)):
                raise SystemExit(f'源文件缺失: {cls}/{fname}')

    missing = [(c, f) for c in CLASSES
               for f in os.listdir(os.path.join(IMG_DIR, c))
               if (c, f) not in seen]
    if missing:
        print(f'警告：{len(missing)} 张源图未出现在清单中，将被跳过。示例：{missing[:3]}')

    if os.path.exists(DST):
        raise SystemExit(f'{DST} 已存在，请先删除再运行（防止重复/混入）')

    for name, items in split.items():
        for cls, fname in items:
            dst_dir = os.path.join(DST, name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(os.path.join(IMG_DIR, cls, fname), os.path.join(dst_dir, fname))

    with open(os.path.join(DST, 'split_manifest.txt'), 'w', encoding='utf-8') as f:
        f.write('# Emotion6 split（来源：task/config/dataset_split/Emotion6，原始论文协议，无 val）\n')
        for name in ('train', 'test'):
            for cls, fname in split[name]:
                f.write(f'{name}\t{cls}\t{fname}\n')

    for name, items in split.items():
        dist = {c: sum(1 for cc, _ in items if cc == c) for c in CLASSES}
        print(f'{name}: 共 {len(items)} 张，分布 {dist}')
    print(f'完成：共复制 {sum(len(v) for v in split.values())} 张到 {DST}')


if __name__ == '__main__':
    main()
