#!/bin/sh

cd ../exper/

python train_fpn_v1.py \
    --arch=vgg_fpn_v1 \
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
    --snapshot_dir=../snapshots/vgg_16_fpn_v1_loc_l3_.2_.5\
    --log_dir=../log/vgg_16_fpn_v1_loc_l3_.2_.5  \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --fpn \
    --loss_w_3=1.0 \
    --loc_branch \
    --loc_start=5 \
    --loc_layer=3 \
    --th_bg=0.2 \
    --th_fg=0.5 \