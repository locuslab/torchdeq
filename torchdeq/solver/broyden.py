'''
References:
    https://github.com/locuslab/deq 
    https://github.com/pv/scipy-work/tree/master/scipy/optimize
'''
import torch
import numpy as np 

from .utils import init_solver_info, batch_flatten, update_state, solver_stat_from_info


__all__ = ['broyden_solver']


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    '''
    ``update`` is the propsoed direction of update.

    Code adapted from scipy.
    '''
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, D)
    # part_Us: (N, D, L_thres)
    # part_VTs: (N, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bd, bdl -> bl', x, part_Us)             # (B, L_thres)
    return -x + torch.einsum('bl, bld -> bd', xTU, part_VTs)    # (B, D)


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (B, D)
    # part_Us: (B, D, L_thres)
    # part_VTs: (B, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bld, bd -> bl', part_VTs, x)            # (B, L_thres)
    return -x + torch.einsum('bdl, bl -> bd', part_Us, VTx)     # (B, D)


def broyden_solver(func, x0, 
        max_iter=50, tol=1e-3, stop_mode='abs', indexing=None,
        LBFGS_thres=None, ls=False, return_final=False, 
        **kwargs):
    """
    Implements the Broyden's method for solving a system of nonlinear equations.

    Args:
        func (callable): The function for which we seek a fixed point.
        x0 (torch.Tensor): The initial guess for the root.
        max_iter (int, optional): The maximum number of iterations. Default: 50.
        tol (float, optional): The convergence criterion. Default: 1e-3.
        stop_mode (str, optional): The stopping criterion. Can be either 'abs' or 'rel'. Default: 'abs'.
        indexing (list, optional): List of iteration indices at which to store the solution. Default: None.
        LBFGS_thres (int, optional): The max_iter for the limited memory BFGS method. None for storing all. Default: None.
        ls (bool, optional): If True, perform a line search at each step. Default: False.
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
        >>> z_star, _, _ = broyden_solver(f, z0)            # Run the Broyden's method
        >>> print((z_star - f(z_star)).norm(p=1))           # Print the numerical error
    """
    # Flatten the initial tensor into (B, *)
    x_est = batch_flatten(x0)
    bsz, dim = x_est.shape

    # Define the function g of (B, D) -> (B, D) for root solving
    g = lambda y: func(y.view_as(x0)).reshape_as(y) - y
    
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    LBFGS_thres = max_iter if LBFGS_thres is None else LBFGS_thres

    gx = g(x_est)
    nstep = 0
    tnstep = 0

    # For fast approximate calculation of inv_jacobian 
    Us = torch.zeros(bsz, dim, LBFGS_thres, dtype=x0.dtype, device=x0.device)   # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, LBFGS_thres, dim, dtype=x0.dtype, device=x0.device)
    update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)                         # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    
    new_objective = 1e8
    
    # Initialize tracking dictionaries for solver statistics
    trace_dict, lowest_dict, lowest_step_dict = init_solver_info(bsz, x0.device)
    nstep, lowest_xest = 0, x0
    
    indexing_list = []

    while nstep < max_iter:
        # Perform a line search and update the state if requested
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)

        # Calculate the absolute and relative differences
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / ((gx + x_est).norm(dim=1) + 1e-9)

        # Update the state based on the new estimate
        lowest_xest = update_state(
                lowest_xest, x_est.view_as(x0), nstep, 
                stop_mode, abs_diff, rel_diff, 
                trace_dict, lowest_dict, lowest_step_dict, return_final
                )

        # Store the solution at the specified index
        if indexing and (nstep+1) in indexing:
            indexing_list.append(lowest_xest)

        new_objective = trace_dict[stop_mode][-1].max() 
        if not return_final and new_objective < tol: break
        
        # Check for lack of progress
        if nstep > 30:
            progress = torch.stack(trace_dict[stop_mode][-30:]).max(dim=1)[0] \
                    / torch.stack(trace_dict[stop_mode][-30:]).min(dim=1)[0]
            if new_objective < 3*tol and progress.max() < 1.3:
                # If there's hardly been any progress in the last 30 steps
                break
        
        # Update the inverses Jacobian approximation using the Broyden's update formula
        part_Us, part_VTs = Us[:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bd,bd->b', vT, delta_gx)[:,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)
    
    # Fill everything up to the max_iter length
    for _ in range(max_iter+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)
 
    info = solver_stat_from_info(stop_mode, lowest_dict, trace_dict, lowest_step_dict)
    return lowest_xest, indexing_list, info