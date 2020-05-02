#!/bin/sh

cd ../exper/

python val_cam_fpn.py \
    --arch=vgg_fpn \
    --gpus=2 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_fpn_cls_l5_1_loc_l4_1_l3_1_s10_.4_.5  \
    --onehot=False \
    --debug \
    --debug_dir=../debug/vgg_16_fpn_cls_l5_1_loc_l4_1_l3_1_s10_.4_.5  \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --threshold=0.9,0.92,0.94,0.96,0.98,0.999 \
    --current_epoch=100 \
    --loss_w_3=1. \
    --loss_w_4=1. \
    --loss_w_5=1. \
    --fpn \
    --loc_branch \