#!/bin/sh

cd ../exper/

python val_DA.py \
    --arch=vgg_DA \
    --gpus=3 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_DA_mixp_DDA_84_ca04 \
    --onehot=False \
    --debug \
    --debug_dir=../debug/vgg_DA_mixp_DDA_84_ca04 \
    --threshold=0.05,0.1,0.15,0.2,0.25 \
    --num_maps=8 \
    --mce \
    --NoHDA \
