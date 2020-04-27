#!/bin/sh

cd ../exper/

python train_cam_fpn.py \
    --arch=vgg_fpn \
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
    --snapshot_dir=../snapshots/vgg_16_bifpn_0_0_1\
    --log_dir=../log/vgg_16_bifpn_0_0_1 \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --loss_w_3=0. \
    --loss_w_4=0. \
    --loss_w_5=1.0 \

