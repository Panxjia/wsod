#!/bin/sh

cd ../exper/

python train_DA_our.py \
    --arch=inception3_CAM345_cos_ori \
    --epoch=100 \
    --lr=0.001 \
    --batch_size=30 \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --resume=False \
    --pretrained_model=google.pth \
    --snapshot_dir=../snapshots/google_DA_mce_NoHDA_NoDDA1_nl_res_dot \
    --log_dir=../log/google_DA_mce_NoHDA_NoDDA1_nl_res_dot \
    --onehot=False \
    --decay_point=80 \
    --seed=0 \
    --mce \
    --NoHDA \
    --NoDDA \
    --non_local \
    --non_local_res \
    --non_local_kernel=-1 \
    --non_local_pf=1 \

