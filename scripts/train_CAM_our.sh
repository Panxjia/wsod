#!/bin/sh

cd ../exper/

python train_cam_our.py \
    --arch=vgg_our \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --snapshot_dir=../snapshots/vgg_16_loc_.2_.5_s10_bin_rep8_woseed \
    --log_dir=../log/vgg_16_loc_.2_.5_s10_bin_rep8_woseed \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --loc_branch \
    --th_bg=0.2 \
    --th_fg=0.5 \
    --loc_start=10 \
    --cls_start=120 \

