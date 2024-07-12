#!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/thumos_i3d.yaml --output pseudolabel

echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudolabel/epoch_060.pth.tar

echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudolabel/epoch_070.pth.tar

echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudolabel/epoch_080.pth.tar

echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudolabel/epoch_090.pth.tar
