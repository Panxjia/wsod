#!/bin/sh

cd ../exper/

python val_fpn_v1.py \
    --arch=vgg_fpn_v1 \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_fpn_v1_loc_l5_s5_memo_.1_.2_lr_.5_a_1_rep2 \
    --onehot=False \
    --debug_dir=../debug/vgg_16_fpn_v1_loc_l5_s5_memo_.1_.2_lr_.5_a_1_rep2 \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --debug \
    --threshold=0.1,0.2,0.3,0.4,0.8,0.9,0.92,0.94,0.96,0.98,0.999 \
    --mce \
    --fpn \
    --current_epoch=100 \
    --vis_th=0.8 \
    --loss_w_5=1.0 \
    --loc_branch \
    --loc_layer=5 \
    --memo \