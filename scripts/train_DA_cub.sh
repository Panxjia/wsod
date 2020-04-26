#!/bin/sh

cd ../exper/

python train_DA_our.py \
    --arch=vgg_DA \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --snapshot_dir=../snapshots/vgg_DA_mixp_DDA_84_bbce_lb_ca02 \
    --log_dir=../log/vgg_DA_mixp_DDA_84_bbce_lb_ca02 \
    --onehot=False \
    --decay_point=80 \
    --cos_alpha=0.02  \
    --num_maps=8 \
    --mixp \
    --seed=0 \
    --bbce \
    --NoHDA \
    --lb \


