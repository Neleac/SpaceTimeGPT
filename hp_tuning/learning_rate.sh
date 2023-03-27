#!/bin/bash

lrs=(
    "5e-5"
    "5e-6"
    "5e-7"
)

for lr in "${lrs[@]}"
do
    python train.py --learning_rate "$lr" --output_dir ../training/learning_rate/"$lr"
done