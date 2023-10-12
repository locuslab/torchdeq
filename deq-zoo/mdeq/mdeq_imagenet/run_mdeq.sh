python3 -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 13333 train_mdeq_imagenet.py \
                --data YOUR_DATA_PATH \
                --save-path ./log_dir \
                --workers 8 \
                --batch-size 64 \
                --learning-rate 0.05 \
                --nesterov \
                --epochs 100 \
                --wd 0.00005 \
                --warmup 0 \
                --f_max_iter 26 \
                --f_solver fixed_point_iter \
                --grad 5 \
                --norm_type weight_norm \
                ${@:1}