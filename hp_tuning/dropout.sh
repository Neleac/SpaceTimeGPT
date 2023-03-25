#!/bin/bash

params=("hidden_dropout_prob" "attention_probs_dropout_prob" "drop_path_rate")
vals=("0.05" "0.1" "0.15")

for param in "${params[@]}"
do
    for val in "${vals[@]}"
    do
        python train.py --"$param" "$val" --output_dir ../training/dropout/"$param"_"$val" || true
    done
done


params=("resid_pdrop" "embd_pdrop" "attn_pdrop")
vals=("0" "0.05" "0.15")

for param in "${params[@]}"
do
    for val in "${vals[@]}"
    do
        python train.py --"$param" "$val" --output_dir ../training/dropout/"$param"_"$val" || true
    done
done