#!/bin/bash

python train.py \
    --seed 17 \
    --batch_size 128 \
    --hidden_size "$1" \
    --learning_rate 0.001 \
    --max_epochs 10 \
    --dataset_path ${HOME}/workspace/ml-data/ \
    --num_workers 8 \
    --cuda \
    || exit 1
