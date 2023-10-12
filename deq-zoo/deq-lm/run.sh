#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=$1 --rdzv_endpoint=localhost:$2 \
    train.py \
        --data data/wikitext-103/ \
        --dataset wt103 \
        --adaptive \
        --div_val 4 \
        --n_layer 3 \
        --d_embed 700 \
        --d_model 700 \
        --n_head 10 \
        --d_head 70 \
        --d_inner 14800  \
        --dropout 0.05 \
        --dropatt 0.0 \
        --optim Adam \
        --lr 0.00025 \
        --warmup_step 8000 \
        --eval_interval 5000 \
        --max_step 150000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --norm_type weight_norm \
        --f_solver fixed_point_iter \
        --f_stop_mode rel \
        --f_tol 1e-3 \
        --f_max_iter 0 \
        --grad 12   \
        --mem       \
        --eval_f_max_iter 12 \
        --global_batch_size 56 \
        ${@:3}
