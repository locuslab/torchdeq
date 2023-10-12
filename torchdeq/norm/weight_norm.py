"""
Weight Normalization from https://arxiv.org/abs/1602.07868
References:
    https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/weight_norm.py
"""
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter


__all__ = ['WeightNorm']


def _norm(p, dim):
    """
    Computes the norm over all dimensions except `dim`.

    Args:
        p (Tensor): Input tensor.
        dim (int): The dimension over which to compute the norm.

    Returns:
        Tensor: The norm of the input tensor.
    """
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)

    
class WeightNorm:
    _target_modules = {
        nn.Linear: ('weight', 0), 
        nn.Conv1d: ('weight', 0), 
        nn.Conv2d: ('weight', 0), 
        nn.Conv3d: ('weight', 0),
        nn.ConvTranspose1d: ('weight', 1),
        nn.ConvTranspose2d: ('weight', 1),
        nn.ConvTranspose3d: ('weight', 1),
        }

    def __init__(self, 
            names, dims,
            learn_scale: bool = True,
            target_norm: float = 1., 
            clip: bool = False,
            clip_value: float = 1.
            ) -> None:
        self.names = names
        self.dims = dims
        
        self.learn_scale = learn_scale
        self.target_norm = target_norm

        self.clip = clip
        self.clip_value = clip_value

    def compute_weight(self, module, name, dim):
        """
        Computes the weight with weight normalization.

        Args:
            module (torch.nn.Module): The module which holds the weight tensor.
            name (str): The name of the weight parameter.
            dim (int): The dimension along which to normalize.

        Returns:
            Tensor: The weight tensor after applying weight normalization.
        """
        weight = getattr(module, name + '_v')
        norm = _norm(weight, dim)

        if self.learn_scale:
            g = getattr(module, name + '_g')
            factor = g / norm
        else:
            factor = self.target_norm / norm
        
        if self.clip:
            factor = torch.minimum(self.clip_value * torch.ones_like(factor), factor)

        return weight * factor

    @classmethod
    def overwrite_kwargs(cls, args, learn_scale, target_norm, clip, clip_value):
        """
        Overwrites certain keyword arguments with their counterparts in `args`.

        Args:
            args (dict): Original keyword arguments.
            learn_scale (bool): Learn the scale factor.
            target_norm (float): Target normalization value.
            clip (bool): Clip the norm to prevent explosion.
            clip_value (float): The maximum norm value when clip is True.

        Returns:
            tuple: Tuple containing the overwritten arguments.
        """
        return not args.get('norm_no_scale', not learn_scale), \
                args.get('norm_target_norm', target_norm), \
                args.get('norm_clip', clip), \
                args.get('norm_clip_value', clip_value)
        
    @classmethod
    def apply(
            cls, module, 
            deq_args=None,
            names=None, 
            dims=None, 
            learn_scale=True,
            target_norm=1., 
            clip=False,
            clip_value=1.):
        """
        Apply weight normalization to a given module.

        Args:
            module (torch.nn.Module): The module to apply weight normalization to.
            deq_args (Union[argparse.Namespace, dict, DEQConfig, Any]): 
                Configuration for the DEQ model. 
                This can be an instance of argparse.Namespace, a dictionary, or an instance of DEQConfig.
                Unknown config will be processed using `get_attr` function.
            names (list or str, optional): The names of the parameters to apply spectral normalization to.
            dims (list or int, optional): The dimensions along which to normalize.
            learn_scale (bool, optional): If true, learn a scale factor during training. Default True.
            target_norm (float, optional): The target norm value. Default 1.
            clip (bool, optional): If true, clip the scale factor. Default False.
            clip_value (float, optional): The value to clip the scale factor to. Default 1.

        Returns:
            WeightNorm: The WeightNorm instance.
        """
        if names is None or dims is None:
            module_type = type(module)
            names, dims = cls._target_modules[module_type]
        
        # Pad args
        if type(names) is str:
            names = [names]

        if type(dims) is int:
            dims = [dims]
    
        assert len(names) == len(dims)
        
        learn_scale, target_norm, clip, clip_value = \
                cls.overwrite_kwargs(deq_args, learn_scale, target_norm, clip, clip_value)
        fn = WeightNorm(names, dims, learn_scale, target_norm, clip, clip_value)

        for name, dim in zip(names, dims):
            weight = getattr(module, name)

            # remove w from parameter list
            del module._parameters[name]

            # add g and v as new parameters and express w as v * min(t, g/||v||)
            if fn.learn_scale:
                module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
            module.register_parameter(name + '_v', Parameter(weight.data))
            setattr(module, name, fn.compute_weight(module, name, dim))
        
        return fn

    def __call__(self, module):
        """
        Recomputes the spectral normalization on the module weights.

        Typically, every time the module is called we need to recompute the weight. However,
        in the case of DEQ, the same weight is shared across layers, and we can save
        a lot of intermediate memory by just recomputing once (at the beginning of first call).

        Args:
            module (torch.nn.Module): The module to apply spectral normalization to.
        """
        for name, dim in zip(self.names, self.dims):
            setattr(module, name, self.compute_weight(module, name, dim))

    def remove(self, module):
        """
        Removes weight normalization from the module.

        Args:
            module (torch.nn.Module): The module to remove weight normalization from.
        """
        for name, dim in zip(self.names, self.dims):
            with torch.no_grad():
                weight = self.compute_weight(module, name, dim)

            delattr(module, name)
            if self.learn_scale:
                del module._parameters[name + '_g']
            del module._parameters[name + '_v']
            module.register_parameter(name, Parameter(weight.data))

