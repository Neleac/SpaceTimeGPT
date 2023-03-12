#!/bin/bash

lrs=(
    "5e-3"
    "1e-3"
    "5e-4"
    "1e-4"
    "5e-5"
    "1e-5"
    "5e-6"
    "1e-6"
    "5e-7"
    "1e-7"
)

for lr in "${lrs[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python train.py "$lr"
done