#!/bin/sh

cd ../exper/

python val_cam_our.py \
    --arch=vgg_our \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/vgg_16_loc_.2_.5_s10_bin_rep9_woseed_fg_first \
    --onehot=False \
    --debug_dir=../debug/vgg_16_loc_.2_.5_s10_bin_rep9_woseed_fg_first \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --debug \
    --threshold=0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.92,0.94,0.96,0.98,0.99,0.999 \
    --mce \
    --current_epoch=100 \
    --loc_branch \