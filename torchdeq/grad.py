"""
The `torchdeq.grad` module offers a factory function, `backward_factory`, 
which is designed to facilitate the customization of various differentiation methods during the backward pass. 

This function is integral to the construction of the backward computational graph in the DEQ class, 
as it is invoked multiple times to generate gradient functors.

While the backward_factory function is a powerful tool, it is generally not recommended for direct use outside of the library. 
Instead, users should primarily interact with the DEQ class via the `torch.core` entry point for most DEQ computations. 
This approach ensures the appropriate and efficient use of the library's features.
"""
import torch
from torch import autograd


__all__ = ['backward_factory']


def make_pair(target, source):
    """
    Aligns the argument sequence between target and source.

    Args:
        target (list): The target list for alignment.
        source (list): The source list for alignment.

    Returns:
        list: The aligned source.

    Raises:
        ValueError: If the length of source is neither 1 nor equal to the length of target.
    """
    if len(target) == len(source):
        return source
    elif len(source) == 1:
        return [source[0] for _ in range(len(target))]
    else:
        raise ValueError('Unable to align the arg squence!')


def backward_factory(
        grad_type='ift',
        hook_ift=False,
        b_solver=None, 
        b_solver_kwargs=dict(),
        sup_gap=-1,
        sup_loc=None,
        tau=1.0,
        **grad_factory_kwargs):
    """
    Factory for the backward pass of implicit deep learning, e.g., DEQ (implicit models), 
    Hamburger (optimization layers), etc.
    This function implements various gradients like Implicit Differentiation (IFT), 1-step Grad and Phantom Grad.

    Implicit Differentiation:
        [2018-ICML] Reviving and Improving Recurrent Back-Propagation

        [2019-NeurIPS] Deep Equilibrium Models
        
        [2019-NeurIPS] Meta-Learning with Implicit Gradients
        
        ...
    1-step Grad & Higher-order Grad:
        [2021-ICLR] Is Attention Better Than Matrix Decomposition? 
        
        [2022-AAAI] JFB: Jacobian-Free Backpropagation for Implicit Networks
        
        [2021-NeurIPS] On Training Implicit Models

        ...
        
    Args:
        grad_type (str, int, optional): 
            Gradient type to use. grad_type should be ``'ift'`` for IFT or an int for PhantomGrad. Default ``'ift'``. 
            Set to ``'ift'`` to enable the implicit differentiation (IFT) mode.
            When passing a number ``k`` to this function, it runs UPG with steps ``k`` and damping factor ``tau``.
        hook_ift (bool, optional): 
            Set to ``True`` to enable an :math:`\Omega(1)` memory (w.r.t. activations) implementation using the Pytorch hook for IFT. 

            Set to ``False`` to enable the :math:`\Omega(2)` memory implementation using ``torch.autograd.Function`` to avoid 
            the (potential) segment fault in older PyTorch versions.

            Note that the ``torch.autograd.Function`` implementation is more stable than this hook in numerics and execution,
            even though they should be conceptually the same.
            For PyTorch version < 1.7.1 on some machines, this :math:`\Omega(1)` hook seems to trigger a segment fault after some training steps.
            This issue is not caused by TorchDEQ but rather due to the hook.remove() call and some interactions between Python and PyTorch. 
            The ``torch.autograd.Function`` implementation also introduces slightly better numerical stability
            when the forward solver introduces some fixed point errors.

            Default ``False``.
        b_solver (str, optional):
            Solver for the IFT backward pass. Default None.
            Supported solvers: ``'anderson'``, ``'broyden'``, ``'fixed_point_iter'``, ``'simple_fixed_point_iter'``.
        b_solver_kwargs (dict, optional):
            Collection of backward solver kwargs, e.g., 
            max_iter (int, optional), max steps for the backward solver, 
            stop_mode (str, optional), criterion for convergence, etc.
            See torchdeq.solver for all kwargs.
        sup_gap (int, optional): 
            The gap for uniformly sampling trajectories from PhantomGrad. Sample every ``sup_gap`` states if ``sup_gap > 0``. Default -1.
        sup_loc (list[int], optional): 
            Specifies trajectory steps or locations in PhantomGrad from which to sample. Default None.
        tau (float, optional):
            Damping factor for PhantomGrad. Default 1.0.
            0.5-0.7 is recommended for MDEQ. 1.0 for DEQ flow.
            For DEQ flow, the gating function in GRU naturally produces adaptive tau values. 
        grad_factory_kwargs:
            Extra arguments are ignored.
    
    Returns:
        callable: 
            A gradient functor for implicit deep learning. The function takes trainer, func and z_pred as arguments
            and returns a list of tensors with the gradient information.        
        
            Args:
                trainer (torch.nn.Module): 
                    the module that employs implicit deep learning.
                func (type): 
                    function that defines the `f` in `z = f(z)`.
                z_pred (torch.Tensor): 
                    latent state to run the backward pass.
                writer (callable, optional): 
                    Callable function to monitor the backward pass. It should accept the solver statistics dictionary as input. Default None.
        
            Returns:
                list[torch.Tensor]: 
                    a list of tensors that tracks the gradient info.
                    These tensors can be directly applied to downstream networks,
                    while all the gradient info will be automatically tracked in the backward pass.
    """
     # IFT grad
    if grad_type == 'ift':
        if hook_ift:
            # IFT via Pytorch hook mechanism
            def hook_ift_grad(trainer, func, z_pred, writer=None, **kwargs):
                z_pred = z_pred.requires_grad_()
                new_z_pred = func(z_pred)        # 1-step grad for df/dtheta
                
                def backward_hook(grad):
                    if trainer.hook is not None:
                        trainer.hook.remove()    # To avoid infinite loop
                    grad_star, _, info = b_solver(
                            lambda y: autograd.grad(new_z_pred, z_pred, y, retain_graph=True)[0] + grad, 
                            torch.zeros_like(grad), **b_solver_kwargs
                            )
                    if writer:
                        writer(info)
                    return grad_star
                trainer.hook = new_z_pred.register_hook(backward_hook)
                
                return [new_z_pred]
            return hook_ift_grad
        else:
            # IFT via torch.autograd.Function
            class IFTGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, func, z_pred, writer):
                    ctx.func, ctx.writer = func, writer
                    ctx.save_for_backward(z_pred.detach())
                    return z_pred

                @staticmethod
                def backward(ctx, grad):
                    func, writer = ctx.func, ctx.writer
                    z_pred, = ctx.saved_tensors

                    h = z_pred.clone().detach().requires_grad_()
                    with torch.enable_grad():
                        f = func(h)
        
                    grad_f = lambda x: autograd.grad(f, h, x, retain_graph=True)[0] + grad
                    grad_star, _, info = b_solver(
                            grad_f, torch.zeros_like(grad), **b_solver_kwargs
                            )
                    if writer:
                        writer(info)

                    return (None, grad_star, None)
            def func_ift(_, func, z_pred, writer=None, **kwargs):
                new_z_pred = func(z_pred)       # 1-step grad for df/dtheta
                return [IFTGrad.apply(func, new_z_pred, writer)]
            return func_ift
    
    # Phantom Grad
    else:
        assert type(grad_type) is int and grad_type >= 1
        n_grad_step = grad_type
        
        if sup_gap > 0:
            def sup_gap_grad_func(_, func, z_pred, **kwargs):
                z_out = []
                for i in range(n_grad_step):
                    z_pred = func(z_pred, tau=tau)
                    if (i+1) % sup_gap == 0:
                        z_out.append(z_pred)

                return z_out
            return sup_gap_grad_func
        elif sup_loc:
            def sup_loc_grad_func(_, func, z_pred, **kwargs):
                z_out = []
                for i in range(n_grad_step):
                    z_pred = func(z_pred, tau=tau)
                    if i+1 in sup_loc:
                        z_out.append(z_pred)
                z_out.append(z_pred)

                return z_out
            return sup_loc_grad_func
        else:
            def grad_func(_, func, z_pred, **kwargs):
                for _ in range(n_grad_step):
                    z_pred = func(z_pred, tau=tau)

                return [z_pred]
            return grad_func
