#!/bin/sh

cd ../exper/

python val_DA.py \
    --arch=inception3_CAM345_cos_ori \
    --gpus=0 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/google_DA_mce_NoHDA_NoDDA1_nl_res_k5 \
    --onehot=False \
    --debug_dir=../debug/google_DA_mce_NoHDA_NoDDA1_nl_res_k5 \
    --restore_from=cub_epoch_100_glo_step_20000.pth.tar \
    --debug \
    --threshold=0.5,0.6,0.7,0.8 \
    --mce \
    --NoHDA \
    --NoDDA \
    --current_epoch=100 \
    --non_local \
    --non_local_res \
    --non_local_kernel=5 \
