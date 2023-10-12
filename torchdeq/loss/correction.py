import torch


__all__ = ['fp_correction', 'register_weight_func']


def _linear(n, k, gamma=0.9, bias=0.0, **kwargs):
    return 1 - (n-k-1) / n * gamma + bias


def _exp(n, k, gamma=0.8, **kwargs):
    return gamma ** (n-k-1)


def _const(n, k, c=1.0):
    return c * torch.ones(n).cuda()


_weight_func_dict = {
        'exp': _exp,
        'linear': _linear,
        'const': _const
        }


def _get_weight_func(name):
    assert name in _weight_func_dict

    return _weight_func_dict[name]


def register_weight_func(name, func):
    """
    Registers a new weight function for fixed point correction.
    
    The weight function should map a pair of integers (n, k) to a float, serving as the weight of loss, 
    where 'n' is the total length of the sequence that converges to the fixed point,
    and 'k' is the order of the current state in the sequence.

    Args:
        name (str): Identifier to associate with the new weight function.
        func (callable): The weight function to register, mapping (n, k) to a float value.

    Raises:
        AssertionError: If ``func`` is not callable.
    """
    assert callable(func)
    _weight_func_dict[name] = func


def _align_list(args):
    max_len = 0
    out_args = []
    for each in args:
        if type(each) not in (list, tuple):
            each = [each]
        out_args.append(each)
        max_len = max(len(each), max_len)
    return out_args, max_len


def _get_idx(args, idx):
    return [each[idx%len(each)] for each in args]


def fp_correction(
        crit, args, 
        weight_func='exp', 
        return_loss_values=False,
        **kwargs
        ):
    """
    Computes fixed-point correction for stabilizing Deep Equilibrium (DEQ) models.
    
    Fixed point correction applies the loss function to a sequence of tensors that converge to the fixed point. 
    The loss value of each tensor tuple is weighted by the weight function. 
    This function automatically aligns the input arguments to be of the same length.

    The currently supported weight functions include ``'const'`` (constant), ``'linear'``, and ``'exp'`` (exponential).

    Args:
        crit (callable): Loss function. Can be the instance of torch.nn.Module or functor.
        args (list or tuple): List of arguments to pass to the criterion.
        weight_func (str, optional): Name of the weight function to use. Default 'exp'.
        return_loss_values (bool, optional): Whether to return the loss values. Default False.
        **kwargs: Additional keyword arguments for the weight function.

    Returns:
        torch.Tensor: The computed loss.
        list[float]: List of individual loss values. Returned only if return_loss_values is set to True.
    
    Examples:
        >>> x = [torch.randn(16, 32, 32) for _ in range(3)]
        >>> y = torch.randn(16, 32, 32)
        >>> mask = torch.rand(16, 32, 32)
        >>> crit = lambda x, y, mask: ((x - y) * mask).abs().mean()
        >>> loss = fp_correction(crit, (x, y, mask))
    """
    args, max_len = _align_list(args)
    weight_func = _get_weight_func(weight_func)

    loss = 0.0
    loss_list = []
    for i in range(max_len):
        i_weight = weight_func(max_len, i, **kwargs)
        i_loss = crit(*_get_idx(args, i))
        loss += i_weight * i_loss

        if return_loss_values:
            loss_list.append(i_loss.item())
    
    if return_loss_values:
        return loss, loss_list
    else:
        return loss

