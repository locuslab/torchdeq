python main.py                  \
    --cfg ./configs/small.yaml  \
    --norm_type weight_norm     \
    --f_solver broyden          \
    --f_max_iter 22             \
    --grad 5                    \
    --tau 0.5                   \
    --exp_id small              \
    ${@:1}
