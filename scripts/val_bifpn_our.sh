#!/bin/sh

cd ../exper/

python val_cam_fpn.py \
    --arch=vgg_fpn \
    --gpus=2 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_fpn_0_0_1 \
    --onehot=False \
    --debug \
    --debug_dir=../debug/vgg_16_fpn_0_0_1_lv3 \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --threshold=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
    --current_epoch=100 \
    --loss_w_3=0. \
    --loss_w_4=0. \
    --loss_w_5=1. \
    --fpn \