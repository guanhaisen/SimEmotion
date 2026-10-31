#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
TASK_TYPE="gqa"
DATA_PATH="./playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl"
IMAGE_FOLDER="./playground/data/eval/gqa/data/images"
ANSWER_DIR="./playground/data/eval/gqa/data/testdev_balanced_predictions.json"
OUTPUT_PATH=${3:-"ece_results.json"}

python -m llava.eval.eval_subtask_calibration \
    --model_path $MODEL_PATH \
    --task_type $TASK_TYPE \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --answer_dir $ANSWER_DIR \
    --output_path ./results/$MODEL_NAME/$TASK_TYPE/$OUTPUT_PATH \
    --plot