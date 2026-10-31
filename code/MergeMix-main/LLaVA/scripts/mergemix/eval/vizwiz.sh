#!/bin/bash
MODEL_PATH=$1
OUTPUT_NAME=$2

python -m llava.eval.model_vqa_loader \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test/test \
    --answers-file ./playground/data/eval/vizwiz/$OUTPUT_NAME\/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/$OUTPUT_NAME\/answers/llava-v1.5-7b.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/$OUTPUT_NAME\/answers_upload/llava-v1.5-7b.json
