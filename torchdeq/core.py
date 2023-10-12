"""
The DEQ models are a class of implicit models that solve for fixed points to make predictions. 
This module provides the core classes and functions for implementing Deep Equilibrium (DEQ) models in PyTorch.

The main classes in this module are `DEQBase`, `DEQIndexing`, and `DEQSliced`. 
`DEQBase` is the base class for DEQ models, and `DEQIndexing` and `DEQSliced` are two specific implementations of DEQ models 
that use different strategies for applying gradients during training.

The module also provides utility functions for creating and manipulating DEQ models, such as `get_deq` for creating a DEQ model based on command line arguments, 
`register_deq` for registering a new DEQ model class, and `reset_deq` for resetting the normalization and dropout layers of a DEQ model.

Example:
    To create a DEQ model, you can use the `get_deq` function:

    >>> deq = get_deq(args)

    To reset the normalization and dropout layers of a DEQ model, you can use the `reset_deq` function:

    >>> deq_layer = DEQLayer(args)          # A Pytorch Module used in the f of z* = f(z*, x).
    >>> reset_deq(deq_layer)
"""

import numpy as np

import torch
import torch.nn as nn

from .solver import get_solver
from .grad import make_pair, backward_factory
from .loss import power_method
from .utils.config import DEQConfig
from .utils.layer_utils import deq_decorator

from .norm import reset_norm
from .dropout import reset_dropout


__all__ = ['get_deq', 'reset_deq', 'register_deq', 'DEQSliced', 'DEQIndexing']


class DEQBase(nn.Module):
    """
    Base class for Deep Equilibrium (DEQ) model.

    This class is not intended to be directly instantiated as the actual DEQ module. 
    Instead, you should create an instance of a subclass of this class.

    If you are looking to implement a new computational graph for DEQ models, you can inherit from this class. 
    This allows you to leverage other components in the library in your implementation.
    
    Args:
        args (Union[argparse.Namespace, dict, DEQConfig, Any], optional): Configuration for the DEQ model. 
            This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
            Unknown config will be processed using `get_attr` function. 
            Priority: ``args`` > ``norm_kwargs``.
            Default None.
        f_solver (str, optional): The forward solver function. Default solver is ``'fixed_point_iter'``.
        b_solver (str, optional): The backward solver function. Default solver is ``'fixed_point_iter'``.
        no_stat (bool, optional): Skips the solver stats computation if True. Default None.
        f_max_iter (int, optional): Maximum number of iterations (NFE) for the forward solver. Default 40.
        b_max_iter (int, optional): Maximum number of iterations (NFE) for the backward solver. Default 40.
        f_tol (float, optional): The forward pass solver stopping criterion. Default 1e-3.
        b_tol (float, optional): The backward pass solver stopping criterion. Default 1e-6.
        f_stop_mode (str, optional): The forward pass fixed-point convergence stop mode. Default ``'abs'``.
        b_stop_mode (str, optional): The backward pass fixed-point convergence stop mode. Default ``'abs'``.
        eval_factor (int, optional): The max iteration for the forward pass at test time, calculated as ``f_max_iter * eval_factor``. Default 1.0.
        eval_f_max_iter (int, optional): The max iteration for the forward pass at test time. Overwrite ``eval_factor`` by an exact number.
        **kwargs: Additional keyword arguments to update the configuration.        
    """
    def __init__(self, 
                 args               =   None,
                 f_solver           =   'fixed_point_iter',
                 b_solver           =   'fixed_point_iter',
                 no_stat            =   None,
                 f_max_iter         =   40,
                 b_max_iter         =   40,
                 f_tol              =   1e-3,
                 b_tol              =   1e-6,
                 f_stop_mode        =   'abs',
                 b_stop_mode        =   'abs',
                 eval_factor        =   1.0,
                 eval_f_max_iter    =   0,
                 **kwargs):
        super(DEQBase, self).__init__()
        
        self.args = DEQConfig(args)
        self.args.update(**kwargs)

        self.f_solver = get_solver(self.args.get('f_solver', f_solver))
        self.b_solver = get_solver(self.args.get('b_solver', b_solver))
        
        if no_stat is None:
            no_stat = self.args.get('f_solver', f_solver) == 'simple_fixed_point_iter'
        self.no_stat = no_stat

        self.f_max_iter = self.args.get('f_max_iter', f_max_iter)
        self.b_max_iter = self.args.get('b_max_iter', b_max_iter)
        
        self.f_tol = self.args.get('f_tol', f_tol)
        self.b_tol = self.args.get('b_tol', b_tol)
        
        self.f_stop_mode = self.args.get('f_stop_mode', f_stop_mode)
        self.b_stop_mode = self.args.get('b_stop_mode', b_stop_mode)
        
        eval_f_max_iter = self.args.get('eval_f_max_iter', eval_f_max_iter)
        eval_factor  = self.args.get('eval_factor', eval_factor)
        self.eval_f_max_iter = eval_f_max_iter if eval_f_max_iter > 0 else int(self.f_max_iter * eval_factor) 

        self.hook = None

    def _sradius(self, deq_func, z_star):
        """
        Estimates the spectral radius using the power method.

        Args:
            deq_func (callable): The DEQ function.
            z_star (torch.Tensor): The fixed point solution.

        Returns:
            float: The spectral radius.
        """
        with torch.enable_grad():
            new_z_star = deq_func(z_star.requires_grad_())
        _, sradius = power_method(new_z_star, z_star, n_iters=100)

        return sradius

    def _solve_fixed_point(
            self, deq_func, z_init,
            f_max_iter=None, 
            solver_kwargs=None,
            **kwargs
            ):
        """
        Solves for the fixed point. Must be overridden in subclasses.

        Args:
            deq_func (callable): The DEQ function.
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            f_max_iter (float, optional): 
                Maximum number of iterations (NFE) for overwriting the solver max_iter in this call. Default None.
            solver_kwargs (dict, optional):
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.

        Raises:
            NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError
    
    def forward(
            self, func, z_init, 
            solver_kwargs=None,
            sradius_mode=False, 
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ. Must be overridden in subclasses.

        Args:
            func (callable): The DEQ function.
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            solver_kwargs (dict, optional): 
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.
            sradius_mode (bool, optional): 
                If True, computes the spectral radius in validation and adds 'sradius' to the ``info`` dictionary. Default False. 
            backward_writer (callable, optional): 
                Callable function to monitor the backward pass. It should accept the solver statistics dictionary as input. Default None.

        Raises:
            NotImplementedError: If the method is not overridden.
        """
        raise NotImplementedError


class DEQIndexing(DEQBase):
    """
    DEQ computational graph that samples fixed point states at specific indices.

    For `DEQIndexing`, it defines a computational graph with tracked gradients by indexing the internal solver
    states and applying the gradient function to the sampled states.
    This is equivalent to attaching the gradient function aside the full solver computational graph. 
    The maximum number of DEQ function calls is defined by ``args.f_max_iter``.

    Args:
        args (Union[argparse.Namespace, dict, DEQConfig, Any], optional): Configuration for the DEQ model. 
            This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
            Unknown config will be processed using `get_attr` function. 
            Priority: ``args`` > ``norm_kwargs``.
            Default None.
        f_solver (str, optional): The forward solver function. Default ``'fixed_point_iter'``.
        b_solver (str, optional): The backward solver function. Default  ``'fixed_point_iter'``.
        no_stat (bool, optional): Skips the solver stats computation if True. Default None.
        f_max_iter (int, optional): Maximum number of iterations (NFE) for the forward solver. Default 40.
        b_max_iter (int, optional): Maximum number of iterations (NFE) for the backward solver. Default 40.
        f_tol (float, optional): The forward pass solver stopping criterion. Default 1e-3.
        b_tol (float, optional): The backward pass solver stopping criterion. Default 1e-6.
        f_stop_mode (str, optional): The forward pass fixed-point convergence stop mode. Default ``'abs'``.
        b_stop_mode (str, optional): The backward pass fixed-point convergence stop mode. Default ``'abs'``.
        eval_factor (int, optional): The max iteration for the forward pass at test time, calculated as ``f_max_iter * eval_factor``. Default 1.0.
        eval_f_max_iter (int, optional): The max iteration for the forward pass at test time. Overwrite ``eval_factor`` by an exact number.
        ift (bool, optional): If true, enable Implicit Differentiation. IFT=Implicit Function Theorem. Default False.
        hook_ift (bool, optional): If true, enable a Pytorch backward hook implementation of IFT. 
            Furthure reduces memory usage but may affect stability. Default False.
        grad (Union[int, list[int], tuple[int]], optional): Specifies the steps of PhantomGrad. 
            It allows for using multiple values to represent different gradient steps in the sampled trajectory states. Default 1.
        tau (float, optional): Damping factor for PhantomGrad. Default 1.0.
        sup_gap (int, optional): 
            The gap for uniformly sampling trajectories from PhantomGrad. Sample every ``sup_gap`` states if ``sup_gap > 0``. Default -1.
        sup_loc (list[int], optional): 
            Specifies trajectory steps or locations in PhantomGrad from which to sample. Default None.
        n_states (int, optional):
            Uniformly samples trajectory states from the solver. 
            The backward passes of sampled states will be automactically tracked. 
            IFT will be applied to the best fixed-point estimation when ``ift=True``, while internal states are tracked by PhantomGrad.
            Default 1. By default, only the best fixed point estimation will be returned. 
        indexing (int, optional):
            Samples specific trajectory states at the given steps in ``indexing`` from the solver. Similar to ``n_states`` but more flexible.
            Default None.
        **kwargs: Additional keyword arguments to update the configuration. 
    """
    def __init__(self, 
                 args           =   None,
                 ift            =   False,
                 hook_ift       =   False,
                 grad           =   1,
                 tau            =   1.0,
                 sup_gap        =   -1,
                 sup_loc        =   None,
                 n_states       =   1,
                 indexing       =   None,
                 **kwargs):
        super(DEQIndexing, self).__init__(args=args, **kwargs)
        
        # Preprocess arguments.
        grad = self.args.get('grad', grad)
        if isinstance(grad, int):
            assert grad > 0, 'The minimal gradient step is 1!'
            grad = [grad]
        
        assert type(grad) in (list, tuple)

        sup_loc  = [] if sup_loc is None else sup_loc
        indexing = [] if indexing is None else indexing

        self.arg_n_states = n_states
        self.arg_indexing = indexing

        '''Define gradient functions through the backward factory.'''

        # First compute the f_max_iter indexing where we add corrections.
        self.indexing = self._compute_f_iter(self.f_max_iter)

        # By default, we use the same phantom grad for all correction losses.
        # You can also set different grad steps a, b, and c for different terms by ``args.grad a b c ...``.
        indexing_pg = make_pair(self.indexing, grad)
        produce_grad = [
                backward_factory(grad_type=pg, 
                                 tau=self.args.get('tau', tau), 
                                 sup_gap=self.args.get('sup_gap', sup_gap),
                                 sup_loc=self.args.get('sup_loc', sup_loc)
                                 ) for pg in indexing_pg
                ]

        # Enabling args.ift will replace the last gradient function by Implicit Differentiation.
        if self.args.get('ift', ift) or self.args.get('hook_ift', hook_ift):
            produce_grad[-1] = backward_factory(
                grad_type='ift', hook_ift=self.args.get('hook_ift', hook_ift), b_solver=self.b_solver,
                b_solver_kwargs=dict(max_iter=self.b_max_iter, tol=self.b_tol, stop_mode=self.b_stop_mode)
                )

        self.produce_grad = produce_grad
    
    def _compute_f_iter(self, f_max_iter):
        """
        Computes the steps for sampling internal solver states.
        Priority: args.n_states > args.indexing.
        Uses args.n_states to uniformly divide the solver forward max_iter if args.n_states is designated.
        Otherwise, uses args.indexing to generate the sample sequence.
        By default, it returns the f_max_iter if no args.n_states or args.indexing apply.
        
        Args:
            f_max_iter (float): Maximum number of iterations (NFE) for the forward solver.

        Returns:
            list[int]: List of solver steps to be sampled.
        """
        arg_n_states = self.args.get('n_states', self.arg_n_states)
        if arg_n_states > 1:
            n_states = max(min(f_max_iter, arg_n_states), 1)
            delta = int(f_max_iter // n_states)
            if f_max_iter % n_states == 0:
                return [(k+1)*delta for k in range(n_states)]
            else:
                return [f_max_iter-(n_states-k-1)*delta for k in range(n_states)]
        else:
            return [*self.args.get('indexing', self.arg_indexing), f_max_iter]

    def _solve_fixed_point(
               self, deq_func, z_init, 
               f_max_iter=None, 
               indexing=None, 
               solver_kwargs=None, 
               **kwargs
               ):
        """
        Solves for the fixed point using the DEQ solver.

        Args:
            deq_func (callable): The DEQ function.
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            f_max_iter (float, optional): Maximum number of iterations (NFE) for overwriting the forward solver max_iter in this call. Default None.
            indexing (list, optional): Trajectory steps/locations for sampling from the DEQ solver. Default None.
            solver_kwargs (dict, optional): 
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
                - torch.Tensor: The fixed point solution.
                - list[torch.Tensor]: Sampled fixed point trajectory according to args.n_states or args.indexing.
                - dict[str, torch.Tensor]: A dict containing solver statistics.
        """
        solver_kwargs = {k:v for k, v in solver_kwargs.items() if k != 'f_max_iter'}
        indexing = indexing if self.training else None

        with torch.no_grad():
            z_star, trajectory, info = self.f_solver(
                    deq_func, x0=z_init, max_iter=f_max_iter,                     # To reuse previous fixed points
                    tol=self.f_tol, stop_mode=self.f_stop_mode, indexing=indexing,
                    **solver_kwargs
                    )

        return z_star, trajectory, info

    def forward(
            self, func, z_init, 
            solver_kwargs=None,
            sradius_mode=False, 
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ.

        This method carries out the forward pass computation for the DEQ model, by solving for the fixed point.
        During training, it also keeps track of the trajectory of the solution. 
        In inference mode, it returns the final fixed point.

        Args:
            func (callable): The DEQ function.
            
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            
            solver_kwargs (dict, optional): 
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.
            sradius_mode (bool, optional): 
                If True, computes the spectral radius in validation and adds ``'sradius'`` to the ``info`` dictionary. Default False. 
            backward_writer (callable, optional): 
                Callable function to monitor the backward pass. It should accept the solver statistics dictionary as input. Default None.

        Returns:
            tuple[list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
    
                - list[torch.Tensor]:
                    | During training, returns the sampled fixed point trajectory (tracked gradients) according to ``n_states`` or ``indexing``.
                    | During inference, returns a list containing the fixed point solution only.
                    
                - dict[str, torch.Tensor]:
                    A dict containing solver statistics in a batch. Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
        """
        deq_func, z_init = deq_decorator(func, z_init, no_stat=self.no_stat)
        
        if solver_kwargs is None:
            solver_kwargs = dict()

        if self.training:
            if type(solver_kwargs.get('f_max_iter', None)) in [int, float]:
                indexing = self._compute_f_iter(solver_kwargs['f_max_iter'])
            else:
                indexing = self.indexing

            _, trajectory, info = self._solve_fixed_point(deq_func, z_init, 
                    f_max_iter=solver_kwargs.get('f_max_iter', self.f_max_iter), indexing=indexing, solver_kwargs=solver_kwargs)
            
            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_pred = deq_func.detach(z_pred)
                z_out += produce_grad(self, deq_func, z_pred, writer=backward_writer)   # See torchdeq.grad for the backward pass
            
            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for the fixed point
            z_star, _, info = self._solve_fixed_point(deq_func, z_init, 
                    f_max_iter=solver_kwargs.get('f_max_iter', self.eval_f_max_iter), solver_kwargs=solver_kwargs)
            
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


class DEQSliced(DEQBase):
    """
    DEQ computational graph that slices the full solver trajectory to apply gradients.

    For `DEQSliced`, it slices the full solver steps into several smaller graphs (w/o grad).
    The gradient function will be applied to the returned state of each subgraph.
    Then a new fixed point solver will resume from the output of the gradient function.
    This is equivalent to inserting the gradient function into the full solver computational graph. 
    The maximum number of DEQ function calls is defined by, for example, ``args.f_max_iter + args.n_states * args.grad``.

    Args:
        args (Union[argparse.Namespace, dict, DEQConfig, Any], optional): Configuration for the DEQ model. 
            This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
            Unknown config will be processed using `get_attr` function. 
            Priority: ``args`` > ``norm_kwargs``.
            Default None.
        f_solver (str, optional): The forward solver function. Default ``'fixed_point_iter'``.
        b_solver (str, optional): The backward solver function. Default  ``'fixed_point_iter'``.
        no_stat (bool, optional): Skips the solver stats computation if True. Default None.
        f_max_iter (int, optional): Maximum number of iterations (NFE) for the forward solver. Default 40.
        b_max_iter (int, optional): Maximum number of iterations (NFE) for the backward solver. Default 40.
        f_tol (float, optional): The forward pass solver stopping criterion. Default 1e-3.
        b_tol (float, optional): The backward pass solver stopping criterion. Default 1e-6.
        f_stop_mode (str, optional): The forward pass fixed-point convergence stop mode. Default ``'abs'``.
        b_stop_mode (str, optional): The backward pass fixed-point convergence stop mode. Default ``'abs'``.
        eval_factor (int, optional): The max iteration for the forward pass at test time, calculated as ``f_max_iter * eval_factor``. Default 1.0.
        eval_f_max_iter (int, optional): The max iteration for the forward pass at test time. Overwrite ``eval_factor`` by an exact number.
        ift (bool, optional): If true, enable Implicit Differentiation. IFT=Implicit Function Theorem. Default False.
        hook_ift (bool, optional): If true, enable a Pytorch backward hook implementation of IFT. 
            Furthure reduces memory usage but may affect stability. Default False.
        grad (Union[int, list[int], tuple[int]], optional): Specifies the steps of PhantomGrad. 
            It allows for using multiple values to represent different gradient steps in the sampled trajectory states. Default 1.
        tau (float, optional): Damping factor for PhantomGrad. Default 1.0.
        sup_gap (int, optional): 
            The gap for uniformly sampling trajectories from PhantomGrad. Sample every ``sup_gap`` states if ``sup_gap > 0``. Default -1.
        sup_loc (list[int], optional): 
            Specifies trajectory steps or locations in PhantomGrad from which to sample. Default None.
        n_states (int, optional):
            Uniformly samples trajectory states from the solver. 
            The backward passes of sampled states will be automactically tracked. 
            IFT will be applied to the best fixed-point estimation when ``ift=True``, while internal states are tracked by PhantomGrad.
            Default 1. By default, only the best fixed point estimation will be returned. 
        indexing (int, optional):
            Samples specific trajectory states at the given steps in ``indexing`` from the solver. Similar to ``n_states`` but more flexible.
            Default None.
        **kwargs: Additional keyword arguments to update the configuration. 
    """
    def __init__(self, 
                 args           =   None,
                 ift            =   False,
                 hook_ift       =   False,
                 grad           =   1,
                 tau            =   1.0,
                 sup_gap        =   -1,
                 sup_loc        =   None,
                 n_states       =   1,
                 indexing       =   None,
                 **kwargs):
        super(DEQSliced, self).__init__(args, **kwargs)
        
        # Preprocess arguments.
        grad = self.args.get('grad', grad)
        if isinstance(grad, int):
            assert grad > 0, 'The minimal gradient step is 1!'
            grad = [grad]
        
        assert type(grad) in (list, tuple)

        sup_loc  = [] if sup_loc is None else sup_loc
        indexing = [] if indexing is None else indexing

        self.arg_n_states = n_states
        self.arg_indexing = indexing

        '''Define gradient functions through the backward factory.'''

        # First compute the f_max_iter indexing where we add corrections.
        self.indexing = self._compute_f_iter(self.f_max_iter)
               
        # By default, we use the same phantom grad for all correction losses.
        # You can also set different grad steps a, b, and c for different terms by ``args.grad a b c ...``.
        indexing_pg = make_pair(self.indexing, grad)
        produce_grad = [
                backward_factory(grad_type=pg, 
                                 tau=self.args.get('tau', tau), 
                                 sup_gap=self.args.get('sup_gap', sup_gap),
                                 sup_loc=self.args.get('sup_loc', sup_loc)
                                 ) for pg in indexing_pg
                ]

        # Enabling args.ift will replace the last gradient function by Implicit Differentiation.
        if self.args.get('ift', ift) or self.args.get('hook_ift', hook_ift):
            produce_grad[-1] = backward_factory(
                grad_type='ift', hook_ift=self.args.get('hook_ift', hook_ift), b_solver=self.b_solver,
                b_solver_kwargs=dict(max_iter=self.b_max_iter, tol=self.b_tol, stop_mode=self.b_stop_mode)
                )

        self.produce_grad = produce_grad
    
    def _compute_f_iter(self, f_max_iter):
        """
        Computes the steps for sampling internal solver states.
        Priority: args.n_states > args.indexing.
        Uses args.n_states to uniformly divide the solver forward max_iter if ``args.n_states`` is designated.
        Otherwise, uses args.indexing to generate the sample sequence.
        By default, it returns the f_max_iter if no args.n_states or args.indexing apply.
        
        Args:
            f_max_iter (float): Maximum number of iterations (NFE) for the forward solver.

        Returns:
            list[int]: List of solver steps to be sampled.
        """
        arg_n_states = self.args.get('n_states', self.arg_n_states)
        if arg_n_states > 1:
            return [int(f_max_iter // arg_n_states) for _ in range(arg_n_states)]
        else:
            return np.diff([0, *self.args.get('indexing', self.arg_indexing), f_max_iter]).tolist()

    def _solve_fixed_point(
            self, deq_func, z_init,
            f_max_iter=None, 
            solver_kwargs=None,
            **kwargs
            ):
        """
        Solves for the fixed point using the DEQ solver.

        Args:
            deq_func (callable): The DEQ function.
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            f_max_iter (float, optional): Maximum number of iterations (NFE) for overwriting the solver max_iter in this call. Default None.
            solver_kwargs (dict, optional): 
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
                - torch.Tensor: The fixed point solution.
                - list[torch.Tensor]: Sampled fixed point trajectory according to args.n_states or args.indexing.
                - dict[str, torch.Tensor]: A dict containing solver statistics.
        """
        solver_kwargs = {k:v for k, v in solver_kwargs.items() if k != 'f_max_iter'}

        with torch.no_grad():
            z_star, _, info = self.f_solver(
                    deq_func, x0=z_init, max_iter=f_max_iter,             # To reuse the previous fixed point
                    tol=self.f_tol, stop_mode=self.f_stop_mode,
                    **solver_kwargs
                    )         
        
        return z_star, info

    def forward(
            self, func, z_star, 
            solver_kwargs=None,
            sradius_mode=False, 
            backward_writer=None,
            **kwargs
            ):
        """
        Defines the computation graph and gradients of DEQ.

        Args:
            func (callable): The DEQ function.
            z_init (torch.Tensor): Initial tensor for fixed point solver.
            solver_kwargs (dict, optional): 
                Additional arguments for the solver used in this forward pass. These arguments will overwrite the default solver arguments. 
                Refer to the documentation of the specific solver for the list of accepted arguments. Default None.
            sradius_mode (bool, optional): 
                If True, computes the spectral radius in validation and adds ``'sradius'`` to the ``info`` dictionary. Default False. 
            backward_writer (callable, optional): 
                Callable function to monitor the backward pass. It should accept the solver statistics dictionary as input. Default None.

        Returns:
            tuple[list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
    
                - list[torch.Tensor]:
                    | During training, returns the sampled fixed point trajectory (tracked gradients) according to ``n_states`` or ``indexing``.
                    | During inference, returns a list containing the fixed point solution only.
                    
                - dict[str, torch.Tensor]:
                    A dict containing solver statistics in a batch. Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
        """
        deq_func, z_star = deq_decorator(func, z_star, no_stat=self.no_stat)
        
        if solver_kwargs is None:
            solver_kwargs = dict()

        if self.training:
            if type(solver_kwargs.get('f_max_iter', None)) in [int, float]:
                indexing = self._compute_f_iter(solver_kwargs['f_max_iter'])
            else:
                indexing = self.indexing

            z_out = []
            for f_max_iter, produce_grad in zip(indexing, self.produce_grad):
                z_star, info = self._solve_fixed_point(deq_func, z_star, f_max_iter=f_max_iter, solver_kwargs=solver_kwargs)
                z_star = deq_func.detach(z_star)
                z_out += produce_grad(self, deq_func, z_star, writer=backward_writer)   # See torchdeq.grad for implementations
                z_star = z_out[-1]                                                      # Add the gradient chain to the solver.

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for the fixed point
            z_star, info = self._solve_fixed_point(deq_func, z_star, 
                    f_max_iter=solver_kwargs.get('f_max_iter', self.eval_f_max_iter), solver_kwargs=solver_kwargs)
            
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


_core = {
    'indexing': DEQIndexing,
    'sliced': DEQSliced,
    }


def register_deq(deq_type, core):
    """
    Registers a user-defined DEQ class for the get_deq function.

    This method adds a new entry to the DEQ class dict with the key as
    the specified DEQ type and the value as the DEQ class.

    Args:
        deq_type (str): The type of DEQ model to register. This will be used as the key in the DEQ class dict.
        core (type): The class defining the DEQ model. This will be used as the value in the DEQ class dict.

    Example:
        >>> register_deq('custom', CustomDEQ)
    """
    _core[deq_type] = core


def get_deq(args=None, **kwargs):        
    """
    Factory function to generate an instance of a DEQ model based on the command line arguments.

    This function returns an instance of a DEQ model class based on the DEQ computational core
    specified in the command line arguments ``args.core``. 
    For example, ``--core indexing`` for DEQIndexing, ``--core sliced`` for DEQSliced, etc.
    
    DEQIndexing and DEQSliced build different computational graphs in training but keep the same for test.

    For `DEQIndexing`, it defines a computational graph with tracked gradients by indexing the internal solver
    states and applying the gradient function to the sampled states.
    This is equivalent to attaching the gradient function aside the full solver computational graph. 
    The maximum number of DEQ function calls is defined by ``args.f_max_iter``.

    For `DEQSliced`, it slices the full solver steps into several smaller graphs (w/o grad).
    The gradient function will be applied to the returned state of each subgraph.
    Then a new fixed point solver will resume from the output of the gradient function.
    This is equivalent to inserting the gradient function into the full solver computational graph. 
    The maximum number of DEQ function calls is defined by, for example, ``args.f_max_iter + args.n_states * args.grad``.

    Args:
        args (Union[argparse.Namespace, dict, DEQConfig, Any]): Configuration specifying the config of the DEQ model. Default None.
            This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
            Unknown config will be processed using `get_attr` function.
        **kwargs: Additional keyword arguments to update the config.

    Returns:
        DEQBase (torch.nn.Module): DEQ module that defines the computational graph from the specified config.
    
    Example:
        To instantiate a DEQ module, you can directly pass keyword arguments to this function:

        >>> deq = get_deq(core='sliced')

        Alternatively, if you're using a config system like `argparse`, you can pass the parsed config as a single object:

        >>> args = argparse.Namespace(core='sliced')
        >>> deq = get_deq(args)
    """
    args = DEQConfig(args)
    args.update(**kwargs)

    core = args.get('core', 'sliced')
    assert core in _core, 'Not registered DEQ class!'
    
    return _core[core](args)
    

def reset_deq(model):
    """
    Resets the normalization and dropout layers of the given DEQ model (usually before each training iteration).

    Args:
        model (torch.nn.Module): The DEQ model to reset.

    Example:
        >>> deq_layer = DEQLayer(args)          # A Pytorch Module used in the f of z* = f(z*, x).
        >>> reset_deq(deq_layer)
    """
    reset_norm(model)
    reset_dropout(model)
