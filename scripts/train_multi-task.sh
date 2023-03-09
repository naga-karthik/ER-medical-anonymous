#!/bin/bash

seeds='1001 171 4242'
seed_save_pred='12345'      # save predictions for this seed

for seed in $seeds; do

    # get starting time:
    start=`date +%s`

    # for ms_brain dataset
    CUDA_VISIBLE_DEVICES=1 python main_pl_multi-task.py -dt ms_brain -m unet-3d -nspv 4 -dep 3 -initf 32 -ps 64 -me 200 -bs 4 -cve 10 -se $seed -lr 1e-4

    # # for scgm dataset
    # CUDA_VISIBLE_DEVICES=3 python main_pl_multi-task.py -dt scgm -m unet-3d -dep 4 -initf 32 -ps 192 -me 250 -bs 2 -cve 10 -se $seed -lr 1e-4
    
    # get ending time:
    end=`date +%s`
    runtime=$((end-start))
    echo "~~~"
    echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
    echo "~~~"
done 

# get starting time:
start=`date +%s`

# save predictions for a single seed
CUDA_VISIBLE_DEVICES=1 python main_pl_multi-task.py -dt ms_brain -m unet-3d -nspv 4 -dep 3 -initf 32 -ps 64 -me 200 -bs 4 -cve 10 -se $seed_save_pred -lr 1e-4 --save_preds
# CUDA_VISIBLE_DEVICES=1 python main_pl_multi-task.py -dt scgm -m unet-3d -dep 4 -initf 32 -ps 192 -me 250 -bs 2 -cve 10 -se $seed_save_pred -lr 1e-4 --save_preds

# get ending time:
end=`date +%s`
runtime=$((end-start))
echo "~~~"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"

echo "---------------------------------------------------------"
echo "Training and testing done"
echo "---------------------------------------------------------"
