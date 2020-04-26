#!/bin/sh

cd ../exper/

python train_DA_our.py \
    --arch=vgg_DA \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=3 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --snapshot_dir=../snapshots/vgg_DA_mixp_bbce_w05_pw03 \
    --log_dir=../log/vgg_DA_mixp_bbce_w05_pw03 \
    --onehot=False \
    --decay_point=80 \
    --cos_alpha=0.01  \
    --num_maps=8 \
    --mixp \
    --bbce  \
    --sup=1 \
    --seed=0 \
    --weight_bce \
    --bce_pos_weight=0.5 \
    --bbce_pos_weight=0.3 \


