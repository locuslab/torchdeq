import torch

from .stat import SolverStat


def init_solver_info(bsz, device, init_loss=1e8):
    """
    Initializes the dictionaries for solver statistics.

    Args:
        bsz (int): Batch size.
        device (torch.device): Device on which the tensors will be allocated.
        init_loss (float, optional): Initial loss value. Default is 1e8.

    Returns:
        tuple: A tuple containing three dictionaries for tracking absolute and relative differences and steps.
    """
    trace_dict = {
            'abs': [torch.tensor(init_loss, device=device).repeat(bsz)],
            'rel': [torch.tensor(init_loss, device=device).repeat(bsz)]
            }
    lowest_dict = {
            'abs': torch.tensor(init_loss, device=device).repeat(bsz),
            'rel': torch.tensor(init_loss, device=device).repeat(bsz)
            }
    lowest_step_dict = {
            'abs': torch.tensor(0., device=device).repeat(bsz),
            'rel': torch.tensor(0., device=device).repeat(bsz),
            }

    return trace_dict, lowest_dict, lowest_step_dict


def batch_masked_mixing(mask, mask_var, orig_var):    
    """
    Applies a mask to ``mask_var`` and the inverse of the mask to ``orig_var``, then sums the result.
    
    Helper function. First aligns the axes of mask to mask_var.
    Then mixes mask_var and orig_var through the aligned mask.
    
    Args:
        mask (torch.Tensor): A tensor of shape (B,).
        mask_var (torch.Tensor): A tensor of shape (B, ...) for the mask to select.
        orig_var (torch.Tensor): A tensor of shape (B, ...) for the reversed mask to select.

    Returns:
        torch.Tensor: A tensor resulting from the masked mixture of ``mask_var`` and ``orig_var``.
    
    Example:
        >>> mask = torch.tensor([True, False])
        >>> mask_var = torch.tensor([[1, 2], [3, 4]])
        >>> orig_var = torch.tensor([[5, 6], [7, 8]])
        >>> result = batch_masked_mixing(mask, mask_var, orig_var)
        >>> result
        tensor([[1, 2],
                [7, 8]])
    """

    if torch.is_tensor(mask_var):
        axes_to_align = len(mask_var.shape) - 1
    elif torch.is_tensor(orig_var):
        axes_to_align = len(orig_var.shape) - 1
    else:
        raise ValueError('Either mask_var or orig_var should be a Pytorch tensor!')
    
    aligned_mask = mask.view(mask.shape[0], *[1 for _ in range(axes_to_align)])

    return aligned_mask * mask_var + ~aligned_mask * orig_var


def batch_flatten(x):
    """
    Flattens a given tensor along all dimensions except the first one.

    If the input tensor has less than 2 dimensions, it reshapes the tensor to have shape (n_elements, 1).

    Args:
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The flattened tensor.

    Example:
        >>> x = torch.rand(2, 3, 4)
        >>> y = batch_flatten(x)
        >>> y.shape
        torch.Size([2, 12])
    """
    return x.flatten(start_dim=1) if x.dim() >= 2 else x.view(x.nelement(), 1)


def update_state(
        lowest_xest, x_est, nstep, 
        stop_mode, abs_diff, rel_diff, 
        trace_dict, lowest_dict, lowest_step_dict, 
        return_final=False
        ):
    """
    Updates the state of the solver during each iteration.

    Args:
        lowest_xest (torch.Tensor): Tensor of lowest fixed point error.
        x_est (torch.Tensor): Current estimated solution.
        nstep (int): Current step number.
        stop_mode (str): Mode of stopping criteria ('rel' or 'abs').
        abs_diff (torch.Tensor): Absolute difference between estimates.
        rel_diff (torch.Tensor): Relative difference between estimates.
        trace_dict (dict): Dictionary to trace absolute and relative differences.
        lowest_dict (dict): Dictionary storing the lowest differences.
        lowest_step_dict (dict): Dictionary storing the steps at which the lowest differences occurred.
        return_final (bool, optional): Whether to return the final estimated value. Default False.

    Returns:
        torch.Tensor: Updated tensor of lowest fixed point error.
    """
    diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
    trace_dict['abs'].append(abs_diff)
    trace_dict['rel'].append(rel_diff)
 
    for mode in ['rel', 'abs']:
        is_lowest = (diff_dict[mode] < lowest_dict[mode]) + return_final
        if mode == stop_mode:
            lowest_xest = batch_masked_mixing(is_lowest, x_est, lowest_xest)
            lowest_xest = lowest_xest.clone().detach() 
        lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
        lowest_step_dict[mode] = batch_masked_mixing(is_lowest, nstep, lowest_step_dict[mode])

    return lowest_xest


def solver_stat_from_info(
        stop_mode, lowest_dict, trace_dict, lowest_step_dict
        ):
    """
    Generates a dict with solver statistics.

    Args:
        stop_mode (str): Mode of stopping criteria ('rel' or 'abs').
        lowest_dict (dict): Dictionary storing the lowest differences.
        trace_dict (dict): Dictionary to trace absolute and relative differences.
        lowest_step_dict (dict): Dictionary storing the steps at which the lowest differences occurred.

    Returns:
        SolverStat: 
            A dict[str, torch.Tensor] containing solver statistics. 
            Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
    """
    return SolverStat.from_solver_info(stop_mode, lowest_dict, trace_dict, lowest_step_dict)


def solver_stat_from_final_step(
        z, fz, nstep=0,
        ):
    """
    Generates a dict with final-step solver statistics.

    Args:
        z (torch.Tensor): Final fixed point estimate.
        fz (torch.Tensor): Function evaluation of final fixed point estimate.
        nstep (int, optional): Total number of steps in the solver. Default 0.

    Returns:
        SolverStat: 
            A dict[str, torch.Tensor] containing solver statistics. 
            Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
    """
    return SolverStat.from_final_step(z, fz, nstep=nstep)


def dummy_solver_stat():
    """
    Generates a dummy solver statistics dict.

    Returns:
        SolverStat: 
            A dict[str, torch.Tensor] containing solver statistics. 
            Please see :class:`torchdeq.solver.stat.SolverStat` for more details.
    """
    return SolverStat()
