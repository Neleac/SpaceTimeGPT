#!/bin/bash

params=("hidden_dropout_prob" "attention_probs_dropout_prob" "drop_path_rate" "resid_pdrop" "embd_pdrop" "attn_pdrop")
vals=("0.25" "0.5" "0.75")

for param in "${params[@]}"
do
    for val in "${vals[@]}"
    do
        python train.py --"$param" "$val" --output_dir ../training/dropout/"$param"_"$val" || true
    done
done