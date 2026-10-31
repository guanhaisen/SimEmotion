# -*- coding: utf-8 -*-
"""artphoto_5fold 数据准备：sad → sadness 重命名 + 完整性校验

用途：artphoto_5fold/ 的 5 折划分（来源 task/config/dataset_split/artphoto_partition，无 val）
    与 main.py ArtPhoto 分支的 classtype（...sadness）差一个命名，本脚本做一次性的
    文件夹重命名（每折 train/test 各 1 个，共 10 个），并对照 split_manifest.txt 校验文件数。

设计：
  - 幂等：sadness 已存在（重复运行）时跳过并提示
  - --dryrun：只打印将要执行的操作，不做任何改动
  - 重命名后自动做完整性校验：每折每类文件数 vs manifest（sad 折算为 sadness）
  - 无 val 是有意的（--no_val 固定轮数协议），不创建 val 目录

用法（SimEmotion 目录、venv 激活下）:
  python prepare_artphoto_5fold.py --dryrun   # 预览
  python prepare_artphoto_5fold.py            # 执行
"""
import argparse
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
FOLD_ROOT = os.path.join(ROOT, 'artphoto_5fold')
FOLDS = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
SPLITS = ['train', 'test']


def load_manifest_counts():
    """从 split_manifest.txt 统计每折每类文件数（sad 计入 sadness）"""
    counts = defaultdict(int)
    path = os.path.join(FOLD_ROOT, 'split_manifest.txt')
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            fold, split, rel = parts
            cls = os.path.dirname(rel)          # 如 'amusement' 或 'sad'
            cls = 'sadness' if cls == 'sad' else cls
            counts[(fold, split, cls)] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description='artphoto_5fold：sad→sadness 重命名 + 完整性校验')
    parser.add_argument('--dryrun', action='store_true', help='只预览，不做改动')
    args = parser.parse_args()

    manifest = load_manifest_counts()
    renamed = 0
    for fold in FOLDS:
        for split in SPLITS:
            src = os.path.join(FOLD_ROOT, fold, split, 'sad')
            dst = os.path.join(FOLD_ROOT, fold, split, 'sadness')
            if os.path.isdir(dst):
                print('[skip] {} 已存在'.format(os.path.relpath(dst, ROOT)))
                if os.path.isdir(src):
                    print('[warn] {} 与 sadness 并存，请人工检查'.format(os.path.relpath(src, ROOT)))
            elif os.path.isdir(src):
                n = len(os.listdir(src))
                print('[rename] {} -> sadness ({} 张)'.format(os.path.relpath(src, ROOT), n))
                if not args.dryrun:
                    os.rename(src, dst)
                renamed += 1
            else:
                print('[warn] {} 与 sadness 均不存在'.format(os.path.relpath(src, ROOT)))

    if args.dryrun:
        print('\n[dryrun] 共将重命名 {} 个文件夹；去掉 --dryrun 重新运行以执行'.format(renamed))
        return

    # ---- 完整性校验：目录实况 vs manifest ----
    print('\n===== 完整性校验 =====')
    if manifest is None:
        print('[warn] 找不到 split_manifest.txt，跳过校验（重命名已完成）')
        return
    all_ok = True
    for fold in FOLDS:
        for split in SPLITS:
            base = os.path.join(FOLD_ROOT, fold, split)
            actual = {}
            if os.path.isdir(base):
                for cls in sorted(os.listdir(base)):
                    p = os.path.join(base, cls)
                    if os.path.isdir(p):
                        actual[cls] = len([f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))])
            for cls in sorted(set(list(actual) + [k[2] for k in manifest if k[0] == fold and k[1] == split])):
                expect = manifest.get((fold, split, cls), 0)
                got = actual.get(cls, 0)
                if expect != got:
                    all_ok = False
                    print('[MISMATCH] {}/{}/{}: 目录 {} vs manifest {}'.format(fold, split, cls, got, expect))
            total = sum(actual.values())
            print('{}/{:5s}: {} 张，8 类齐全={}'.format(fold, split, total, len(actual) == 8))
    print('\n校验结果：{}'.format('全部一致' if all_ok else '存在不一致，请检查上方 MISMATCH 行'))


if __name__ == '__main__':
    main()
