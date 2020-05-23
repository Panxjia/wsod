#!/bin/sh

cd ../exper/

python val_fpn_v1.py \
    --arch=vgg_fpn_v1 \
    --gpus=3 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_fpn_v1_sel_.25_.35_s10_l5_1._l4_1._sep_l3_1._sep_mask \
    --onehot=False \
    --debug_dir=../debug/vgg_16_fpn_v1_sel_.25_.35_s10_l5_1._l4_1._sep_l3_1._sep_mask  \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --debug \
    --threshold=0.1,0.2,0.3,0.4,0.5,0.6 \
    --mce \
    --fpn \
    --current_epoch=100 \
    --vis_th=0.2 \
    --loss_w_3=1.0 \
    --loss_w_4=1.0 \
    --loss_w_5=1.0 \
    --com_feat \