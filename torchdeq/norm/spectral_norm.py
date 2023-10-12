"""
Spectral Normalization from https://arxiv.org/abs/1802.05957
References:
    https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py
"""
import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter


__all__ = ['SpectralNorm']


def _view_back_dim(tensor, tgt_shape, dims):
    """
    Given the input tensor shape, this function operates
    the inverse of dimensionality reduction in sum, mean, etc.

    Args:
        tensor (torch.Tensor): Input tensor.
        tgt_shape (list or tuple): Target shape for the output tensor.
        dims (int or list of ints, optional): The dimensions to be reshaped.

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    if dims is None:
        dims = [i for i in range(len(tgt_shape))] 
    elif type(dims) is int:
        dims = [dims]
        
    to_shape = []
    for i, size in enumerate(tgt_shape):
        size = 1 if i not in dims else size
        to_shape.append(size)
        
    return tensor.reshape(to_shape)


class SpectralNorm(object):
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
            clip_value: float = 1.,
            n_power_iterations: int = 1, 
            eps: float = 1e-12) -> None:
        self.names = names
        self.dims = dims
        
        self.learn_scale = learn_scale
        self.target_norm = target_norm

        self.clip = clip
        self.clip_value = clip_value

        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def _reshape_weight_to_matrix(self, weight, dim):
        """
        Reshapes the weight tensor into a matrix.

        Args:
            weight (torch.Tensor): The weight tensor.
            dim (int): The dimension along which to reshape.

        Returns:
            torch.Tensor: The reshaped weight tensor.
        """
        weight_mat = weight
        if dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(dim,
                                            *[d for d in range(weight_mat.dim()) if d != dim])
        height = weight_mat.shape[0]
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module, do_power_iteration, name, dim):
        """
        Computes the weight with spectral normalization.

        Args:
            module (torch.nn.Module): The module which holds the weight tensor.
            do_power_iteration (bool): If true, do power iteration for approximating singular vectors.
            name (str): The name of the weight parameter.
            dim (int): The dimension along which to normalize.

        Returns:
            torch.Tensor: The computed weight tensor.
        """
        weight = getattr(module, name + '_orig')
        u = getattr(module, name + '_u')
        v = getattr(module, name + '_v')
        weight_mat = self._reshape_weight_to_matrix(weight, dim)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py#L46
                    # on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))

        if self.learn_scale:
            g = getattr(module, name + '_g')
            factor = g / sigma
        else:
            factor = self.target_norm / sigma
        
        if self.clip:
            factor = torch.minimum(self.clip_value * torch.ones_like(factor), factor)

        return weight * factor
    
    @classmethod
    def overwrite_kwargs(cls, args, learn_scale, target_norm, clip, clip_value, n_power_iterations):
        """
        Overwrites certain keyword arguments with their counterparts in `args`.

        Args:
            args (argparse.Namespace): The namespace which holds the arguments.
            learn_scale (bool): Learn the scale factor.
            target_norm (float): Target normalization value.
            clip (bool): Clip the norm to prevent explosion.
            clip_value (float): The maximum norm value when clip is True.
            n_power_iterations (int): Number of power iterations.

        Returns:
            tuple: Tuple containing the overwritten arguments.
        """
        return not args.get('norm_no_scale', not learn_scale), \
                args.get('norm_target_norm', target_norm), \
                args.get('norm_clip', clip), \
                args.get('norm_clip_value', clip_value), \
                args.get('sn_n_power_iters', n_power_iterations)
        
    @classmethod
    def apply(
            cls, module, 
            deq_args=None,
            names=None, 
            dims=None, 
            learn_scale=True,
            target_norm=1., 
            clip=False,
            clip_value=1.,
            n_power_iterations=1, 
            eps=1e-12):
        """
        Applies spectral normalization to a given module.

        Args:
            module (torch.nn.Module): The module to apply spectral normalization to.
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
            n_power_iterations (int, optional): The number of power iterations to perform. Default 1.
            eps (float, optional): A small constant for numerical stability. Default 1e-12.

        Returns:
            SpectralNorm: The SpectralNorm instance.
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

        learn_scale, target_norm, clip, clip_value, n_power_iterations = \
                cls.overwrite_kwargs(deq_args, learn_scale, target_norm, clip, clip_value, n_power_iterations)
        fn = SpectralNorm(names, dims, learn_scale, target_norm, clip, clip_value, n_power_iterations, eps)
        
        for name, dim in zip(names, dims):
            weight = module._parameters[name]

            with torch.no_grad():
                weight_mat = fn._reshape_weight_to_matrix(weight, dim)

            h, w = weight_mat.shape
            # randomly initialize `u` and `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

            delattr(module, name)
            module.register_parameter(name + "_orig", weight)
            # We still need to assign weight back as name because all sorts of
            # things may assume that it exists, e.g., when initializing weights.
            # However, we can't directly assign as it could be an nn.Parameter and
            # gets added as a parameter. Instead, we register weight.data as a plain
            # attribute.
            setattr(module, name, weight.data)
            if fn.learn_scale:
                g_data = _view_back_dim(target_norm * torch.ones(h), weight.shape, dim)
                module.register_parameter(name + '_g', Parameter(g_data))
            module.register_buffer(name + "_u", u)
            module.register_buffer(name + "_v", v)
            
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
            setattr(module, name, self.compute_weight(module, module.training, name, dim))
    
    def remove(self, module):
        """
        Removes spectral normalization from the module.

        Args:
            module (torch.nn.Module): The module to remove spectral normalization from.
        """
        for name, dim in zip(self.names, self.dims):
            with torch.no_grad():
                weight = self.compute_weight(module, False, name, dim)
            delattr(module, name)
            delattr(module, name + '_u')
            delattr(module, name + '_v')
            if self.learn_scale:
                delattr(module, name + '_g')
            delattr(module, name + '_orig')
            module.register_parameter(name, Parameter(weight.detach()))


