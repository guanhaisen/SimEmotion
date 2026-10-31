#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
TASK_TYPE="pope"
DATA_PATH="./playground/data/eval/pope/llava_pope_test.jsonl"
IMAGE_FOLDER="./playground/data/eval/pope/val2014"
ANSWER_DIR="./playground/data/eval/pope/coco"
OUTPUT_PATH=${3:-"ece_results.json"}

python -m llava.eval.eval_subtask_calibration \
    --model_path $MODEL_PATH \
    --task_type $TASK_TYPE \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --answer_dir $ANSWER_DIR \
    --output_path ./results/$MODEL_NAME/$TASK_TYPE/$OUTPUT_PATH \
    --plot