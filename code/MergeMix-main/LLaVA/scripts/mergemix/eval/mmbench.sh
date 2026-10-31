#!/bin/bash
MODEL_PATH=$1
OUTPUT_NAME=$2

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/$OUTPUT_NAME\/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/$OUTPUT_NAME\/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/$OUTPUT_NAME\/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/$OUTPUT_NAME\/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
