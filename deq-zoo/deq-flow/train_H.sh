#!/bin/bash

python -u main.py --total_run 1 --start_run 1 --name deq-flow-H-120k-C-36-6-1 \
    --stage chairs --validation chairs kitti \
    --gpus 0 1 2 --num_steps 120000 --eval_interval 20000 \
    --batch_size 12 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 \
    --f_max_iter 36 --f_solver simple_fixed_point_iter \
    --n_states 6 --grad 1 \
    --huge

python -u main.py --total_run 1 --start_run 1 --name deq-flow-H-120k-T-40-2-3 \
    --stage things --validation sintel kitti \
    --restore_name deq-flow-H-120k-C-36-6-1 \
    --gpus 0 1 2 --num_steps 120000 \
    --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 \
    --f_max_iter 40 --f_solver simple_fixed_point_iter \
    --n_states 2 --grad 3 \
    --huge --all_grad 

