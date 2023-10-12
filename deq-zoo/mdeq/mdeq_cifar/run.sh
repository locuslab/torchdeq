python main.py                  \
    --cfg ./configs/large.yaml  \
    --norm_type weight_norm     \
    --f_solver broyden          \
    --f_max_iter 18             \
    --grad 5                    \
    --tau 0.7                   \
    --exp_id large              \
    ${@:1}

