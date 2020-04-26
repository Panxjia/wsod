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
    --snapshot_dir=../snapshots/vgg_DA_mixp_mce_sel0001_20a \
    --log_dir=../log/vgg_DA_mixp_mce_sel0001_20a \
    --onehot=False \
    --decay_point=80 \
    --cos_alpha=0.01  \
    --num_maps=8 \
    --mixp \
    --mce  \
    --seed=0 \
    --cls_th=0.0001 \

