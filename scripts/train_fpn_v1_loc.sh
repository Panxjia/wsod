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
    --snapshot_dir=../snapshots/vgg_16_fpn_v1_loc_l5_s5_memo_.1_.2_lr_.5_a_1_rep3 \
    --log_dir=../log/vgg_16_fpn_v1_loc_l5_s5_memo_.1_.2_lr_.5_a_1_rep3 \
    --onehot=False \
    --decay_point=80 \
    --mce \
    --fpn \
    --loss_w_5=1.0 \
    --loc_branch \
    --loc_start=5 \
    --loc_layer=5 \
    --th_bg=0.1 \
    --th_fg=0.2 \
    --memo \
    --memo_lr=0.5 \
    --memo_alpha=1 \