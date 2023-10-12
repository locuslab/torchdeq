


def add_deq_args(parser):
    """
    Decorate the commonly used argument parser with arguments used in TorchDEQ.

    Args:
        parser (argparse.Namespace): Command line arguments.    
    """
    # Solver
    parser.add_argument('--f_solver', default='fixed_point_iter', type=str, 
            help='forward solver to use. supported solvers: anderson, broyden, fixed_point_iter, simple_fixed_point_iter')
    parser.add_argument('--b_solver', default='fixed_point_iter', type=str, choices=['anderson', 'broyden', 'fixed_point_iter', 'simple_fixed_point_iter'],
            help='backward solver to use. supported solvers: anderson, broyden, fixed_point_iter, simple_fixed_point_iter')
    parser.add_argument('--f_max_iter', type=int, default=40, help='max number of function evaluations (NFE) for the forward pass solver')
    parser.add_argument('--b_max_iter', type=int, default=40, help='max number of function evaluations (NFE) for the backward pass solver')
    parser.add_argument('--f_tol', type=float, default=1e-3, help='forward pass solver stopping criterion')
    parser.add_argument('--b_tol', type=float, default=1e-6, help='backward pass solver stopping criterion')
    parser.add_argument('--f_stop_mode', type=str, default='abs', help='forward pass fixed-point convergence stop mode')
    parser.add_argument('--b_stop_mode', type=str, default='abs', help='backward pass fixed-point convergence stop mode')
    parser.add_argument('--eval_factor', type=float, default=1.0, help='factor to scale up the f_max_iter at test for better convergence.')
    parser.add_argument('--eval_f_max_iter', type=int, default=0, help='directly set the f_max_iter at test.')

    # Norm
    parser.add_argument('--norm_type', default='weight_norm', type=str,
                        help='Normalizations for DEQ, using the form of [W <- W * min(norm_clip_value, target_norm / norm)], \
                                choices=[none, weight_norm, spectral_norm]')
    parser.add_argument('--norm_no_scale', action='store_true', help='Remove the learnable target_norm from normalization.')
    parser.add_argument('--norm_clip', action='store_true', help='Clip the scale value if (target_norm / sigma) > norm_clip_value.')
    parser.add_argument('--norm_clip_value', default=1.0, type=float, help='Upper Bound for the (re)normalization factor.')
    parser.add_argument('--norm_target_norm', default=1.0, type=float, help='Pre-defined target norm when not learning the target_norm.')
    parser.add_argument('--sn_n_power_iters', default=1, type=int, help='Number of power iterations to estimate the spectral radius.')

    # Training
    parser.add_argument('--core', default='sliced', type=str, help='training container for DEQ. choices=[indexing, sliced]')
    parser.add_argument('--ift', action='store_true', help='use implicit differentiation.')
    parser.add_argument('--hook_ift', action='store_true', help='use a hook function of O(1) memory complexity for IFT.')
    parser.add_argument('--n_states', type=int, default=1, help='number of loss terms (uniform spaced, 1 + fixed point correction).')
    parser.add_argument('--indexing', type=int, nargs='+', default=[], help='index solver states for fixed point correction.')
    parser.add_argument('--gamma', type=float, default=0.8, help='control the strength of fixed point correction. See loss.py in torchdeq.')
    parser.add_argument('--grad', type=int, nargs='+', default=[1], help='steps of Phantom Grad.')
    parser.add_argument('--tau', type=float, default=1.0, help='damping factor for unrolled Phantom Grad')
    parser.add_argument('--sup_gap', type=int, default=-1, help='sampling gap along the trajectories by Phantom Grad.')
    parser.add_argument('--sup_loc', type=int, nargs='+', default=[],  help='sampling location along the trajectories by Phantom Grad.')

    # Regularization
    parser.add_argument('--jac_loss_weight', type=float, default=0.0,
                    help='jacobian regularization loss weight (default to 0)')
    parser.add_argument('--jac_loss_freq', type=float, default=0.0,
                    help='the frequency of applying the jacobian regularization (default to 0)')
    parser.add_argument('--jac_incremental', type=int, default=0,
                    help='if positive, increase jac_loss_weight by 0.1 after this many steps')

    # Monitoring
    parser.add_argument('--sradius_mode', action='store_true', help='monitor the spectral radius during validation')


