#!/bin/bash
MODEL_PATH=$1
OUTPUT_NAME=$2

python -m llava.eval.model_vqa_science \
    --model-path $MODEL_PATH \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/$OUTPUT_NAME/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/$OUTPUT_NAME/answers/llava-v1.5-7b.jsonl \
    --output-file ./playground/data/eval/scienceqa/$OUTPUT_NAME/answers/llava-v1.5-7b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/$OUTPUT_NAME/answers/llava-v1.5-7b_result.json