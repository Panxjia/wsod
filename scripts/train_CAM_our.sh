#!/bin/sh

cd ../exper/

python train_cam_our.py \
    --arch=vgg_our \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=3 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=vgg16.pth \
    --seed=0 \
    --snapshot_dir=../snapshots/vgg_16_loc_.2_.5_s10\
    --log_dir=../log/vgg_16_loc_.2_.5_s10  \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --loc_branch \
    --loc_start=10 \
    --th_bg=0.2 \
    --th_fg=0.5 \

