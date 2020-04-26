#!/bin/sh

cd ../exper/

python val_DA.py \
    --arch=vgg_DA \
    --gpus=3 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_DA_mixp_mce_sel0001_20a \
    --onehot=False \
    --debug \
    --debug_dir=../debug/vgg_mixp_mce_sel0001_20a \
    --threshold=0.05,0.1,0.15,0.2,0.25 \
    --mce \
