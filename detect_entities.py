"""给 emotion_dataset 重新生成实体词（cate info），对齐论文 DE 模块。

规则: 全景分割 score>0.5 的类别词 -> 小写、去 -merged/-other 后缀、多词取首词
      （已与作者 cate info 的 131 词逐一核对，完全同构）
输出: D:/Coding/Python_Projects/SimEmotion/entity_words/FI/<图片名stem>.txt
用法: python detect_entities.py [--limit 5] [--overwrite]
"""
import argparse, re, sys, traceback
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 防少量坏图中断全量任务

SRC = Path(r"d:\VibeCoding\Trae\project\task\emotion_dataset")
DST = Path(r"D:\Coding\Python_Projects\SimEmotion\entity_words\FI")
MODEL = r"D:\Coding\Python_Projects\SimEmotion\model_weights\mask2former-swin-tiny"  # 本地权重，离线可用
CLASSES = ["amusement", "anger", "awe", "contentment", "disgust", "excitement", "fear", "sadness"]
SCORE_TH = 0.5
FAIL_LOG = Path(r"d:\VibeCoding\Trae\project\task\entity_failures.txt")


def clean_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"-merged$", "", n)
    n = re.sub(r"-other$", "", n)
    return n.split(" ")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（小样本测试用）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的 txt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to(device).eval()
    id2label = model.config.id2label
    print("model:", MODEL, "| labels:", len(id2label))

    images = []
    for ct in CLASSES:
        images += sorted((SRC / ct).glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    print("images to process:", len(images))

    done = skipped = failed = 0
    shown = 0
    with open(FAIL_LOG, "a", encoding="utf-8") as fail_f, tqdm(images) as bar:
        for img_path in bar:
            out_path = DST / (img_path.stem + ".txt")
            if out_path.exists() and not args.overwrite:   # 断点续跑
                skipped += 1
                continue
            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                res = processor.post_process_panoptic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                segs = sorted(
                    (s for s in res["segments_info"] if s["score"] > SCORE_TH),
                    key=lambda s: s["score"], reverse=True,
                )
                words = list(dict.fromkeys(clean_name(str(id2label[s["label_id"]])) for s in segs))
                DST.mkdir(parents=True, exist_ok=True)
                out_path.write_text(" ".join(words), encoding="utf-8")  # 无实体则写空文件
                done += 1
                if shown < 10:
                    print(f"{img_path.parent.name}/{img_path.name} -> {' '.join(words) or '(空)'}")
                    shown += 1
            except Exception:
                failed += 1
                fail_f.write(img_path.as_posix() + "\n" + traceback.format_exc() + "\n")
                if failed <= 5:
                    print("FAIL:", img_path, file=sys.stderr)
    print(f"done={done} skipped={skipped} failed={failed}")
    print("output dir:", DST)


if __name__ == "__main__":
    main()