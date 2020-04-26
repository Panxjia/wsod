#!/bin/sh

cd ../exper/

python train_DA_our.py \
    --arch=vgg_DA \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --snapshot_dir=../snapshots/vgg_DA_mixp_sc00001_73_child \
    --log_dir=../log/vgg_DA_mixp_sc00001_73_child \
    --onehot=False \
    --decay_point=80 \
    --cos_alpha=0.01  \
    --num_maps=8 \
    --mixp \
    --mce \
    --seed=0 \
    --sc \
    --sc_alpha=0.00001 \
    --sc_old=0.7 \
    --sc_new=0.3 \

