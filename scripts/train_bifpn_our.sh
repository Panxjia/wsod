#!/bin/sh

cd ../exper/

python train_cam_fpn.py \
    --arch=vgg_fpn \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=2 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_fpn_cls_l5_loc_l3_l4_s10_.2_.5 \
    --log_dir=../log/vgg_16_fpn_cls_l5_loc_l3_l4_s10_.2_.5 \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --loss_w_3=1. \
    --loss_w_4=1. \
    --loss_w_5=1. \
    --fpn \
    --loc_branch \
    --loc_start=10 \
    --th_bg=0.2 \
    --th_fg=0.5 \


