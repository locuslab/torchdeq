"""
The `torchdeq.norm` module provides a set of tools for managing normalization in Deep Equilibrium Models (DEQs). 
It includes factory functions for applying, resetting, and removing normalization, 
as well as for registering new normalization types and modules. 

The module also provides classes for specific types of normalization, such as `WeightNorm` and `SpectralNorm`.

Example:
    To apply normalization to a model, call this `apply_norm` function:

    >>> apply_norm(model, 'weight_norm', filter_out=['embedding'])

    To reset the all normalization within a DEQ model, call this `reset_norm` function:

    >>> reset_norm(model)

    To remove the normalization of a DEQ model, call `remove_norm` function:

    >>> remove_norm(model)

    To register a user-defined normalization type, call `register_norm` function:

    >>> register_norm('custom_norm', CustomNorm)

    To register a new module for a user-define normalization, call `register_norm_module` function:

    >>> register_norm_module(Conv2d, 'custom_norm', 'weight', 0)
"""


from .base_norm import (apply_norm, register_norm, register_norm_module,
                        remove_norm, reset_norm)
