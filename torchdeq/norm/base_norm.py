from ..utils.config import DEQConfig

from .weight_norm import WeightNorm
from .spectral_norm import SpectralNorm


__all__ = ['apply_norm', 'reset_norm', 'remove_norm',
           'register_norm', 'register_norm_module']


_norm_class = {
        'weight_norm': WeightNorm,
        'spectral_norm': SpectralNorm,
        }


def register_norm(norm_type, norm_class):
    """
    Registers a user-defined normalization class for the apply_norm function.

    This function adds a new entry to the Norm class dict with the key as
    the specified ``norm_type`` and the value as the ``norm_class``.

    Args:
        norm_type (str): The type of normalization to register. This will be used as the key in the Norm class dictionary.
        norm_class (type): The class defining the normalization. This will be used as the value in the Norm class dictionary.

    Example:
        >>> register_norm('custom_norm', CustomNorm)
    """
    _norm_class[norm_type] = norm_class


def register_norm_module(module_class, norm_type, names='weight', dims=0):
    """
    Registers a to-be-normed module for the user-defined normalization class in the `apply_norm` function.

    This function adds a new entry to the _target_modules attribute of the specified normalization class in 
    the _norm_class dictionary. The key is the module class and the value is a tuple containing the attribute name 
    and dimension over which to compute the norm.

    Args:
        module_class (type): Module class to be indexed for the user-defined normalization class.
        norm_type (str): The type of normalization class that the module class should be registered for.
        names (str, optional): Attribute name of ``module_class`` for the normalization to be applied. Default ``'weight'``.
        dims (int, optional): Dimension over which to compute the norm. Default 0.

    Example:
        >>> register_norm_module(Conv2d, 'custom_norm', 'weight', 0)
    """
    _norm_class[norm_type]._target_modules[module_class] = (names, dims)


def _is_skip_prefix(name, prefix_filter_out):
    """
    Helper function to check if a module name starts with any string in the filter_out list.

    Args:
        name (str): Name of the module.
        prefix_filter_out (list of str): List of string prefixes to filter out.

    Returns:
        bool: True if the module name starts with any string in the filter_out list, False otherwise.
    """
    for skip_name in prefix_filter_out:
        if name.startswith(skip_name):
            return True
    
    return False


def _is_skip_name(name, filter_out):
    """
    Helper function to check if a given module name contains any string in the filter_out list.

    Args:
        name (str): Name of the module.
        filter_out (list of str): List of strings to be filtered out.

    Returns:
        bool: True if the module name contains any string in the filter_out list, False otherwise.
    """
    for skip_name in filter_out:
        if skip_name in name:
            return True
    
    return False


def apply_norm(model, norm_type='weight_norm', prefix_filter_out=None, filter_out=None, args=None, **norm_kwargs):
    """
    Auto applies normalization to all weights of a given layer based on the ``norm_type``.
    
    The currently supported normalizations include ``'weight_norm'``, ``'spectral_norm'``, and ``'none'`` (No Norm applied).
    Skip the weights whose name contains any string of ``filter_out`` or starts with any of ``prefix_filter_out``.

    Args:
        model (torch.nn.Module): Model to apply normalization.
        norm_type (str, optional): Type of normalization to be applied. Default is ``'weight_norm'``.
        prefix_filter_out (list or str, optional): 
            List of module weights prefixes to skip out when applying normalization. Default is None.
        filter_out (list or str, optional): 
            List of module weights names to skip out when applying normalization. Default is None.
        args (Union[argparse.Namespace, dict, DEQConfig, Any]): Configuration for the DEQ model. 
            This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
            Unknown config will be processed using ``get_attr`` function.
            Priority: ``args`` > ``norm_kwargs``.
            Default is None.
        norm_kwargs: Keyword arguments for the normalization layer.

    Raises:
        AssertionError: If the ``norm_type`` is not registered.

    Example:
        >>> apply_norm(model, 'weight_norm', filter_out=['embedding'])
    """
    args = DEQConfig(args)
    args.update(**norm_kwargs)

    norm_type = args.get('norm_type', norm_type)
    if norm_type == 'none':
        return

    assert norm_type in _norm_class, 'Not registered norm type!'
    Norm = _norm_class[norm_type]

    if isinstance(prefix_filter_out, str):
        prefix_filter_out = [prefix_filter_out]
    
    if isinstance(filter_out, str):
        filter_out = [filter_out]

    for name, module in model.named_modules():
        if prefix_filter_out and _is_skip_prefix(name, prefix_filter_out):
            continue 
        if filter_out and _is_skip_name(name, filter_out):
            continue 

        if type(module) in Norm._target_modules:
            module._deq_norm = Norm.apply(module, deq_args=args, **norm_kwargs)


def reset_norm(model):
    """
    Auto resets the normalization of a given DEQ model.

    Args:
        model (torch.nn.Module): Model to reset normalization.

    Example:
        >>> reset_norm(model)
    """
    for module in model.modules():
        if hasattr(module, '_deq_norm'):
            module._deq_norm(module)


def remove_norm(model):
    """
    Removes the normalization of a given DEQ model.

    Args:
        model (torch.nn.Module): A DEQ model to remove normalization.

    Example:
        >>> remove_norm(model)
    """
    for module in model.modules():
        if hasattr(module, '_deq_norm'):
            module._deq_norm.remove(module)
            del module._deq_norm


