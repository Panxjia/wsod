#!/bin/sh

cd ../exper/

python val_cam_fpn.py \
    --arch=vgg_fpn \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_s10_.2_.5_rep4/  \
    --onehot=False \
    --debug \
    --debug_dir=../debug/vgg_16_s10_.2_.5_rep4/  \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --threshold=0.8,0.9,0.92,0.95,0.97,0.9999,1.0 \
    --current_epoch=100 \
    --loc_branch \
    --loss_w_5=1.0 \