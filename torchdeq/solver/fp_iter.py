from .utils import (batch_flatten, dummy_solver_stat,
                    solver_stat_from_final_step,
                    solver_stat_from_info, init_solver_info,
                    update_state)

__all__ = ['fixed_point_iter', 'simple_fixed_point_iter']


def fixed_point_iter(func, x0, 
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None, 
        tau=1.0, return_final=False, 
        **kwargs):
    """
    Implements the fixed-point iteration solver for solving a system of nonlinear equations.
    
    Args:
        func (callable): The function for which we seek a fixed point.
        x0 (torch.Tensor): The initial guess for the root.
        max_iter (int, optional): The maximum number of iterations. Default: 50.
        tol (float, optional): The convergence criterion. Default: 1e-3.
        stop_mode (str, optional): The stopping criterion. Can be either 'abs' or 'rel'. Default: 'abs'.
        indexing (list, optional): List of iteration indices at which to store the solution. Default: None.
        tau (float, optional): Damping factor. It is used to control the step size in the direction of the solution. Default: 1.0.
        return_final (bool, optional): If True, run all steps and returns the final solution instead of the one with smallest residual. Default: False.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: Fixed point solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]: 
                A dict containing solver statistics in a batch.
                Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
    
    Examples:
        >>> f = lambda z: torch.cos(z)                  # Function for which we seek a fixed point
        >>> z0 = torch.tensor(0.0)                      # Initial estimate
        >>> z_star, _, _ = fixed_point_iter(f, z0)      # Run Fixed Point iterations.
        >>> print((z_star - f(z_star)).norm(p=1))       # Print the numerical error
    """
    # Check input batch size
    bsz = x0.shape[0] if x0.dim() >= 2 else x0.nelement()

    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'

    # Initialize solver statistics
    trace_dict, lowest_dict, lowest_step_dict = init_solver_info(bsz, x0.device)
    lowest_xest = x0

    indexing_list = []
    
    fx = x = x0
    for k in range(max_iter):
        x = fx
        fx = tau * func(x) + (1 - tau) * x
        
        # Calculate the absolute and relative differences# Update the state based on the new estimate
        gx = fx - x
        abs_diff = batch_flatten(gx).norm(dim=1)
        rel_diff = abs_diff / (batch_flatten(fx).norm(dim=1) + 1e-9)

        # Update the state based on the new estimate
        lowest_xest = update_state(
                lowest_xest, fx, k+1, 
                stop_mode, abs_diff, rel_diff, 
                trace_dict, lowest_dict, lowest_step_dict, return_final
                )

         # If indexing is enabled, store the solution at the specified indices
        if indexing and (k+1) in indexing:
            indexing_list.append(lowest_xest)

        # If the difference is smaller than the given tolerance, terminate the loop early
        if not return_final and trace_dict[stop_mode][-1].max() < tol:
            for _ in range(max_iter-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    info = solver_stat_from_info(stop_mode, lowest_dict, trace_dict, lowest_step_dict)
    return lowest_xest, indexing_list, info


def simple_fixed_point_iter(func, x0, 
        max_iter=50, tau=1.0,
        indexing=None, 
        **kwargs):
    """
    Implements a simplified fixed-point solver for solving a system of nonlinear equations.
    
    Speeds up by removing statistics monitoring.

    Args:
        func (callable): The function for which the fixed point is to be computed.
        x0 (torch.Tensor): The initial guess for the fixed point.
        max_iter (int, optional): The maximum number of iterations. Default: 50.
        tau (float, optional): Damping factor to control the step size in the solution direction. Default: 1.0.
        indexing (list, optional): List of iteration indices at which to store the solution. Default: None.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: The approximate solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]: 
                A dummy dict for solver statistics. All values are initialized as -1 of tensor shape (1, 1).

    Examples:
        >>> f = lambda z: torch.cos(z)                      # Function for which we seek a fixed point
        >>> z0 = torch.tensor(0.0)                          # Initial estimate
        >>> z_star, _, _ = simple_fixed_point_iter(f, z0)   # Run fixed point iterations
        >>> print((z_star - f(z_star)).norm(p=1))           # Print the numerical error
    """
    indexing_list = []
    
    fx = x = x0
    for k in range(max_iter):
        x = fx
        fx = func(x, tau=tau)

         # If indexing is enabled, store the solution at the specified indices
        if indexing and (k+1) in indexing:
            indexing_list.append(fx)
    lowest_xest = fx

    info = solver_stat_from_final_step(x, fx, max_iter)
    return lowest_xest, indexing_list, info
