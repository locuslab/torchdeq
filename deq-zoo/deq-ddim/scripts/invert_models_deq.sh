#!/bin/bash

########### CIFAR10 ##################
#### DEQ #############################

for i in $(seq 1 100)
do
    python main.py \
    --config cifar10_ls_opt.yml --model DiffusionInversion --exp cifar10-orig-exp --image_folder cifar10-inverse-t1000-parallel-f-15 --doc cifar10-torchdeq-f-15 \
    --ls_opt --timesteps 1000 --ni --method default --lambda1 1 --lambda2 0 --lambda3 0 \
    --f_solver anderson --f_max_iter 15 --tau 0.1 --use_wandb --no_augmentation --use_pretrained --skip_type quad --seed $i
done


