#!/bin/bash

MODEL_PATH=$1
MODEL_NAME=$2
TASK_TYPE="sqa"
DATA_PATH="./playground/data/eval/scienceqa/llava_test_CQM-A.json"
IMAGE_FOLDER="./playground/data/eval/scienceqa/images/test"
ANSWER_DIR=None
OUTPUT_PATH=${3:-"ece_results.json"}

python -m llava.eval.eval_subtask_calibration \
    --model_path $MODEL_PATH \
    --task_type $TASK_TYPE \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --answer_dir $ANSWER_DIR \
    --output_path ./results/$MODEL_NAME/$TASK_TYPE/$OUTPUT_PATH \
    --plot