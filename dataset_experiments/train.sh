#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py
#CUDA_VISIBLE_DEVICES=0 python train.py --random_frames true
CUDA_VISIBLE_DEVICES=0 python train.py --random_captions true
CUDA_VISIBLE_DEVICES=0 python train.py --random_frames true --random_captions true
