#!/bin/bash

seeds='1001 171 4242' 
seed_save_pred='12345'      # save predictions for this seed

# if training on ms_brain
# NOTE: actual center names are hidden for the sake of anonymity
center_names='BW KA MI RE NI MO UC AM'

# if training on scgm
# center_names='UC UN VA ZR'

for seed in $seeds; do
    for center_name in $center_names; do
        # for ms_brain dataset
        CUDA_VISIBLE_DEVICES=0 python main_pl_single-task.py -m unet-3d -dt ms_brain -cn $center_name -nspv 4 -dep 3 -initf 32 -ps 64 -me 200 -bs 4 -cve 10 -se $seed -lr 1e-4

        # # for scgm dataset
        # CUDA_VISIBLE_DEVICES=0 python main_pl_single-task.py -m unet-3d -dt scgm -cn $center_name -dep 4 -initf 32 -ps 192 -me 250 -bs 2 -cve 10 -se $seed -lr 1e-4 
    done
done 

# save predictions for a single seed
for center_name in $center_names; do
    CUDA_VISIBLE_DEVICES=3 python main_pl_single-task.py -dt ms_brain -cn $center_name -nspv 4 -dep 3 -initf 32 -ps 64 -me 200 -bs 4 -cve 10 -se $seed_save_pred -lr 1e-4 --save_preds
done

echo "---------------------------------------------------------"
echo "Training and testing done"
echo "---------------------------------------------------------"
