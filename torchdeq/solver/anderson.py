'''
References:
    https://github.com/locuslab/deq 
    https://github.com/pv/scipy-work/tree/master/scipy/optimize
'''
import torch

from .utils import init_solver_info, batch_flatten, update_state, solver_stat_from_info


__all__ = ['anderson_solver']


def anderson_solver(func, x0, 
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None,
        m=6, lam=1e-4, tau=1.0, return_final=False, 
        **kwargs):
    """
    Implements the Anderson acceleration for fixed-point iteration.

    Anderson acceleration is a method that can accelerate the convergence of fixed-point iterations. It improves
    the rate of convergence by generating a sequence that converges to the fixed point faster than the original
    sequence.

    Args:
        func (callable): The function for which we seek a fixed point.
        x0 (torch.Tensor): Initial estimate for the fixed point.
        max_iter (int, optional): Maximum number of iterations. Default: 50.
        tol (float, optional): Tolerance for stopping criteria. Default: 1e-3.
        stop_mode (str, optional): Stopping criterion. Can be 'abs' for absolute or 'rel' for relative. Default: 'abs'.
        indexing (None or list, optional): Indices for which to store and return solutions. If None, solutions are not stored. Default: None.
        m (int, optional): Maximum number of stored residuals in Anderson mixing. Default: 6.
        lam (float, optional): Regularization parameter in Anderson mixing. Default: 1e-4.
        tau (float, optional): Damping factor. It is used to control the step size in the direction of the solution. Default: 1.0.
        return_final (bool, optional): If True, returns the final solution instead of the one with smallest residual. Default: False.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: Fixed point solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]: 
                A dict containing solver statistics in a batch.
                Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
            
    Examples:
        >>> f = lambda z: 0.5 * (z + 2 / z)                 # Function for which we seek a fixed point
        >>> z0 = torch.tensor(1.0)                          # Initial estimate
        >>> z_star, _, _ = anderson_solver(f, z0)           # Run Anderson Acceleration
        >>> print((z_star - f(z_star)).norm(p=1))           # Print the numerical error
    """
    # Wrap the input function to ensure the same shape
    f = lambda x: func(x.view_as(x0)).reshape_as(x) 
    
    # Flatten the input tensor into (B, *)
    x0_flat = batch_flatten(x0)
    bsz, dim = x0_flat.shape

    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    # Initialize tensors to store past values and their images under the fixed-point function
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)

    # Initialize the first two values for X and F
    X[:,0], F[:,0] = x0_flat, f(x0_flat)
    X[:,1], F[:,1] = F[:,0], f(F[:,0])
    
    # Initialize tensors for the Anderson mixing process
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict, lowest_dict, lowest_step_dict = init_solver_info(bsz, x0.device)
    lowest_xest = x0

    indexing_list = []

    for k in range(2, max_iter):
        # Apply the Anderson mixing process to compute a new estimate
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]

        X[:,k%m] = tau * (alpha[:,None] @ F[:,:n])[:,0] + (1-tau) * (alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m])
        
        # Calculate the absolute and relative differences
        gx = F[:,k%m] - X[:,k%m]
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / (F[:,k%m].norm(dim=1) + 1e-9)

        # Update the state based on the new estimate
        lowest_xest = update_state(
                lowest_xest, F[:,k%m].view_as(x0), k+1,
                stop_mode, abs_diff, rel_diff, 
                trace_dict, lowest_dict, lowest_step_dict, return_final
                )

        # Store the solution at the specified index
        if indexing and (k+1) in indexing:
            indexing_list.append(lowest_xest)

        # If the difference is smaller than the given tolerance, terminate the loop early
        if not return_final and trace_dict[stop_mode][-1].max() < tol:
            for _ in range(max_iter-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    
    # If no solution was stored during the iteration process, store the final estimate
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    X = F = None
 
    info = solver_stat_from_info(stop_mode, lowest_dict, trace_dict, lowest_step_dict)
    return lowest_xest, indexing_list, info