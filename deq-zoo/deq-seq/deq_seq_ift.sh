#!/bin/bash

echo 'Run training (DEQ-Transformer)...'
python train.py \
    --data data/wikitext-103/ \
    --dataset wt103 \
    --adaptive \
    --div_val 4 \
    --n_layer 2 \
    --eval_n_layer 24 \
    --d_embed 700 \
    --d_model 700 \
    --n_head 10 \
    --d_head 70 \
    --d_inner 48000 \
    --dropout 0.05 \
    --dropatt 0.0 \
    --optim Adam \
    --lr 0.00025 \
    --warmup_step 16000 \
    --eval-interval 5000 \
    --max_step 300000 \
    --tgt_len 150 \
    --mem_len 150 \
    --eval_tgt_len 150 \
    --norm_type weight_norm \
    --ift \
    --f_solver broyden \
    --b_solver broyden \
    --f_stop_mode rel \
    --b_stop_mode rel \
    --f_tol 1e-3 \
    --b_tol 1e-3 \
    --f_max_iter 35 \
    --b_max_iter 35 \
    --jac_loss_weight 0.0 \
    --jac_loss_freq 0.0 \
    --jac_incremental 0 \
    --batch_size 56 \
    --gpu0_bsz 14 \
    --multi_gpu \
    ${@:1}