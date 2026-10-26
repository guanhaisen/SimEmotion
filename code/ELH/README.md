# To Err like Human: Affective Bias-Inspired Measures for Visual Emotion Recognition Evaluation

This repository contains the implementation of our paper "To Err like Human: Affective Bias-Inspired Measures for Visual Emotion Recognition Evaluation", which introduces a new evaluation metric for Visual Emotion Recognition (VER).

## Dataset Preparation
1. Download the FI dataset
2. Download the EmoSet dataset [from here](https://github.com/JingyuanYY/EmoSet)
3. Organize the dataset structure as follows:

```
dataset_root/
    ├── train/
    │   ├── Amusement/
    │   ├── Anger/
    │   └── ...
    └── test/
        ├── Amusement/
        ├── Anger/
        └── ...
```

## Training

To train the model, use the following command:

```bash
python train.py --dataset FI --path '/path/to/dataset' --network resnet50
```
