#!/bin/sh

cd ../exper/

for i in {10..100..10}
do
python val_DA.py \
    --arch=inception3_CAM345_cos_ori \
    --gpus=1 \
    --dataset=cub \
    --img_dir=../data/CUB_200_2011/images \
    --num_classes=200 \
    --snapshot_dir=../snapshots/google_DA_mixp_mce_NoHDA_NoDDA_bin_16 \
    --onehot=False \
    --debug_dir=../debug/google_DA_mixp_mce_NoHDA_NoDDA_bin_16 \
    --restore_from=cub_epoch_${i}_glo_step_$[i*200].pth.tar \
    --debug \
    --threshold=0.15,0.2,0.3,0.4,0.5 \
    --mce \
    --NoHDA \
    --bin_cls \
    --current_epoch=${i}
done