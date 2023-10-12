#!/bin/bash

python -u main.py --eval --name deq-flow-B-things --stage things \
    --validation kitti sintel --restore_ckpt val_checkpoints/deq-flow-B-things-1.pth --gpus 0 \
    --f_max_iter 40 --f_solver fixed_point_iter  \
    --eval_factor 1.5

python -u main.py --eval --name deq-flow-B-things --stage things \
    --validation kitti sintel --restore_ckpt val_checkpoints/deq-flow-B-things-2.pth --gpus 0 \
    --f_max_iter 40 --f_solver fixed_point_iter  \
    --eval_factor 1.5

python -u main.py --eval --name deq-flow-H-things --stage things \
    --validation kitti sintel --restore_ckpt val_checkpoints/deq-flow-H-things-test-1x.pth --gpus 0 \
    --f_max_iter 40 --f_solver fixed_point_iter  \
    --eval_factor 1.5 --huge

python -u main.py --eval --name deq-flow-H-things --stage things \
    --validation kitti sintel --restore_ckpt val_checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --f_max_iter 40 --f_solver fixed_point_iter  \
    --eval_factor 1.5 --huge

python -u main.py --eval --name deq-flow-H-things --stage things \
    --validation kitti sintel --restore_ckpt val_checkpoints/deq-flow-H-things-test-3x.pth --gpus 0 \
    --f_max_iter 40 --f_solver fixed_point_iter  \
    --eval_factor 3.0 --huge

