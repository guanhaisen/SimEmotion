# -*- coding: utf-8 -*-
"""emotion6_official_split 数据准备：官方 Emotion6 划分落盘 + 完整性校验

用途：dataset/Emotion6/（6 类 × 330 = 1,980 张）按官方划分
    task/config/dataset_split/Emotion6/{trainimg_label,testimg_label}.txt 落盘为
    emotion6_official_split/{train,test}/{anger,disgust,fear,joy,sadness,surprise}/
    （train 1,386 / test 594，每类 231/99，70/30，六类完全均衡；无 val——官方二划分）

标签映射（已由文件夹归属 100% 验证，非字母序）：
    0=surprise 1=joy 2=fear 3=sadness 4=disgust 5=anger

设计：
  - 复制不移动（源目录保持完整）；manifest 记录 split<TAB>类/文件 标签
  - 幂等：已存在的目标文件跳过并提示
  - --dryrun：只统计将要执行的复制，不做改动
  - 落盘后自动校验：每折每类文件数 vs 划分清单、train/test 无交集、无缺失无多余

用法（SimEmotion 目录、venv 激活下）:
  python prepare_emotion6_official.py --dryrun   # 预览
  python prepare_emotion6_official.py            # 执行
"""
import argparse
import os
import shutil

CFG = r'd:\VibeCoding\Trae\project\task\config\dataset_split\Emotion6'
DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', 'Emotion6')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotion6_official_split')
LISTS = {'train': 'trainimg_label.txt', 'test': 'testimg_label.txt'}
CLASSES = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def load_lists():
    """读取官方划分清单：行格式 '类名/文件名 标签'；返回 {relpath: (split, label)}"""
    entries = {}
    for split, fname in LISTS.items():
        path = os.path.join(CFG, fname)
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    print('[warn] 跳过异常行: {!r}'.format(line))
                    continue
                rel, label = parts[0], parts[1]
                if rel in entries:
                    print('[warn] 重复条目（后写覆盖先写）: {}'.format(rel))
                entries[rel] = (split, label)
    return entries


def main():
    parser = argparse.ArgumentParser(description='Emotion6 官方划分落盘 + 校验')
    parser.add_argument('--dryrun', action='store_true', help='只统计将要执行的复制，不做改动')
    args = parser.parse_args()

    entries = load_lists()
    total = len(entries)
    n_train = sum(1 for s, _ in entries.values() if s == 'train')
    n_test = total - n_train
    print('清单条目：{}（train {} / test {}）'.format(total, n_train, n_test))

    # ---- 复制 ----
    copied = skipped = missing_src = 0
    for rel, (split, _) in sorted(entries.items()):
        src = os.path.join(DS, rel.replace('/', os.sep))
        dst = os.path.join(OUT, split, rel.replace('/', os.sep))
        if not os.path.isfile(src):
            print('[MISSING] 源文件不存在: {}'.format(src))
            missing_src += 1
            continue
        if os.path.isfile(dst):
            skipped += 1
            continue
        if not args.dryrun:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1
    if args.dryrun:
        print('\n[dryrun] 将复制 {} 个文件（已存在 {} 个将跳过，源缺失 {}）'
              .format(copied, skipped, missing_src))
        return

    # ---- manifest ----
    os.makedirs(OUT, exist_ok=True)
    manifest = os.path.join(OUT, 'split_manifest.txt')
    with open(manifest, 'w', encoding='utf-8') as f:
        f.write('# Emotion6 官方划分（来源 task/config/dataset_split/Emotion6，无 val，二划分 train/test）\n')
        f.write('# 标签映射：0=surprise 1=joy 2=fear 3=sadness 4=disgust 5=anger（非字母序）\n')
        for rel, (split, label) in sorted(entries.items()):
            f.write('{}\t{} {}\n'.format(split, rel, label))
    print('manifest 已写入: {}'.format(manifest))

    # ---- 完整性校验 ----
    print('\n===== 完整性校验 =====')
    all_ok = True
    # 1) 落盘实况 vs 清单
    disk = {}
    for split in LISTS:
        for cls in CLASSES:
            d = os.path.join(OUT, split, cls)
            n = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]) if os.path.isdir(d) else 0
            disk[(split, cls)] = n
    expect = {}
    for rel, (split, _) in entries.items():
        cls = rel.split('/')[0]
        expect[(split, cls)] = expect.get((split, cls), 0) + 1
    for split in LISTS:
        for cls in CLASSES:
            mark = '' if disk[(split, cls)] == expect.get((split, cls), 0) else '  [MISMATCH 预期 {}]'.format(expect.get((split, cls), 0))
            if mark:
                all_ok = False
            print('{}/{}: {} 张{}'.format(split, cls, disk[(split, cls)], mark))
    # 2) train/test 无交集（相对路径层面天然无交集；校验文件名级以防误复制）
    names = {}
    overlap = 0
    for rel, (split, _) in entries.items():
        fn = os.path.basename(rel)
        names.setdefault(fn, set()).add(split)
    overlap = sum(1 for v in names.values() if len(v) > 1)
    # 文件名跨折重复是正常的（每类各自 1.jpg..330.jpg），只校验 同类同名 不跨折
    clsname = {}
    overlap_same_cls = 0
    for rel, (split, _) in entries.items():
        cls, fn = rel.split('/')[0], os.path.basename(rel)
        clsname.setdefault((cls, fn), set()).add(split)
    overlap_same_cls = sum(1 for v in clsname.values() if len(v) > 1)
    print('train/test 交集（同类同名跨折）: {}'.format(overlap_same_cls))
    if overlap_same_cls:
        all_ok = False
    print('\n校验结果：{}'.format('全部一致' if all_ok and missing_src == 0 else '存在不一致，请检查上方 MISMATCH/MISSING 行'))


if __name__ == '__main__':
    main()
