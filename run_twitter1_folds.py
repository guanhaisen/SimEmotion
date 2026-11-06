"""run_twitter1_folds.py — Twitter1 五折循环驱动 v2（官方 val=test 协议，参数化 loss）

逐折调用 main.py（每折独立进程，GPU 显存自然隔离），逐折自动归档带 loss 标记的日志，
全部跑完后聚合双口径（官方口径 = FinalTest = test 峰值；严格口径 = Test@Epoch:24 末轮）。

用法（项目 venv、项目根目录）：
  python run_twitter1_folds.py                       # 默认 ce，全部 5 折
  python run_twitter1_folds.py --loss vpt            # vpt 臂，全部 5 折（自动加 --vpt_l 32）
  python run_twitter1_folds.py --loss paco 2 3 4 5   # 指定 loss + 折号（断点续跑）
日志归档：result/log_twitter1_{loss}_fold{i}.txt（已存在则跳过归档但正常训练）
"""
import glob
import os
import re
import shutil
import subprocess
import sys

# ---------- 参数解析：位置参数=折号；--loss X；'--' 后原样透传 main.py ----------
argv = sys.argv[1:]
FOLDS, LOSS, EXTRA = [], 'ce', []
i = 0
while i < len(argv):
    if argv[i] == '--loss':
        LOSS = argv[i + 1]; i += 2
    elif argv[i] == '--':
        EXTRA = argv[i + 1:]; break
    else:
        FOLDS.append(int(argv[i])); i += 1
FOLDS = FOLDS or [1, 2, 3, 4, 5]

PY = sys.executable
CWD = r'D:\Coding\Python_Projects\SimEmotion'
RESULT = os.path.join(CWD, 'result')
BASE_CMD = [PY, 'main.py', '--datasetname', 'Twitter1', '--loss', LOSS,
            '--modelname', 'ViT-B/16', '--batch_size', '32',
            '--num_epochs', '25', '--seed', '43']
if LOSS == 'vpt':
    BASE_CMD += ['--vpt_l', '32']
BASE_CMD += EXTRA

os.chdir(CWD)
archived = set()
for i in FOLDS:
    cmd = BASE_CMD + ['--datadir', f'twitter1_fold{i}']
    print(f'\n===== fold{i} ({LOSS}) start =====')
    r = subprocess.run(cmd, cwd=CWD)
    if r.returncode != 0:
        print(f'fold{i} FAILED (exit {r.returncode})，中止后续折')
        sys.exit(r.returncode)
    # 逐折归档：找最新的、尚未归档的 Twitter1 日志
    logs = sorted(glob.glob(os.path.join(CWD, 'log', 'ViT-B-16_Twitter1*.txt')),
                  key=os.path.getmtime, reverse=True)
    for lp in logs:
        if lp not in archived:
            dst = os.path.join(RESULT, f'log_twitter1_{LOSS}_fold{i}.txt')
            if os.path.exists(dst):
                print(f'  警告：{dst} 已存在，本次日志未归档（可能重跑）')
            else:
                shutil.copy2(lp, dst)
            archived.add(lp)
            break
    print(f'===== fold{i} done =====')

# ---------- 聚合（从带标记的归档日志读取） ----------
official, strict, rows = [], [], []
for i in FOLDS:
    lp = os.path.join(RESULT, f'log_twitter1_{LOSS}_fold{i}.txt')
    if not os.path.exists(lp):
        print(f'警告：缺 {lp}，该折未计入聚合'); continue
    txt = open(lp, encoding='utf-8', errors='ignore').read()
    m_fin = re.findall(r'Test Loss: [\d.]+ Acc: ([\d.]+) macroF1: ([\d.]+)', txt)
    m_e24 = re.findall(r'Test@Epoch:24 Acc: ([\d.]+)', txt)
    off = float(m_fin[-1][1]) * 100 if m_fin else None
    stc = float(m_e24[-1]) * 100 if m_e24 else None
    official.append(off); strict.append(stc)
    rows.append((i, off, stc))

print(f'\n========== Twitter1 五折聚合（loss={LOSS}, seed 43, 25ep, 官方 val=test 协议） ==========')
print(f'{"fold":<6}{"官方口径 FinalTest":>20}{"严格口径 ep24":>16}')
for i, off, stc in rows:
    print(f'{i:<6}{off:>19.2f}%{stc:>15.2f}%')
import statistics as st
ok_o = [a for a in official if a is not None]
ok_s = [a for a in strict if a is not None]
if len(ok_o) == len(FOLDS) and len(ok_s) == len(FOLDS):
    print(f'\n官方口径（val=test 选模型，论文可比）: {st.mean(ok_o):.2f} ± {st.stdev(ok_o):.2f}')
    print(f'严格口径（末轮）                   : {st.mean(ok_s):.2f} ± {st.stdev(ok_s):.2f}')
else:
    print(f'\n警告：折数不齐（official {len(ok_o)}/{len(FOLDS)}, strict {len(ok_s)}/{len(FOLDS)}）')
