#!/bin/sh

cd ../exper/

python train_fpn_v1.py \
    --arch=vgg_fpn_v1 \
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
    --snapshot_dir=../snapshots/vgg_16_fpn_v1_sel_.25_.35_s10_l5_1._l4_1._sep_l3_1._sep_mask\
    --log_dir=../log/vgg_16_fpn_v1_sel_.25_.35_s10_l5_1._l4_1._sep_l3_1._sep_mask  \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --fpn \
    --loss_w_4=1.0 \
    --loss_w_3=1.0 \
    --loss_w_5=1.0 \
    --erase \
    --erase_start=10 \
    --erase_th_l5=0.25 \
    --erase_th_l4=0.35 \
    --var_erase \
    --l5_red \