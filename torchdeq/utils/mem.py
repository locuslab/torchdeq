import torch


__all__ = ['mem_gc']


def filter_input(in_args):
    """
    Filters the input arguments to distinguish tensors that require gradient.

    Args:
        in_args (tuple): Input arguments.

    Returns:
        tuple: A tuple consisting of the processed input arguments.
        tuple: Tensors from the input arguments that require gradients.
        tuple: The indices of the tensors that require gradients in the original input arguments.
    """
    forward_args = ()
    grad_args = ()
    grad_idx = ()
    for i, arg in enumerate(in_args):
        if torch.is_tensor(arg):
            arg_ready = arg.detach().requires_grad_(arg.requires_grad)
            forward_args += (arg_ready,)
            if arg.requires_grad:
                grad_args += (arg_ready,)
                grad_idx += (i,)
        else:
            forward_args += (arg,)
    
    return forward_args, grad_args, grad_idx 


def filter_out(out, out_grad):
    """
    Filters the output and its gradient to retain tensor pairs.

    Args:
        out (tuple): Tuple containing the output.
        out_grad (tuple): Tuple containing the gradient of the output.

    Returns:
        tuple: Tuple containing tensors from the output.
        tuple: Tuple containing the corresponding gradients of output tensors.
    """
    out_tensor = ()
    out_grad_tensor = ()
    for out_v, out_v_grad in zip(out, out_grad):
        if torch.is_tensor(out_v_grad):
            out_tensor += (out_v,)
            out_grad_tensor += (out_v_grad,)
    
    return out_tensor, out_grad_tensor 
   

def reset_grad(grad, in_args, grad_idx):
    """
    Reorders the gradients to their original positions in the input arguments.

    Args:
        grad (tuple): Tuple containing gradients.
        in_args (tuple): Tuple containing input arguments.
        grad_idx (tuple): Tuple containing indices of gradients.

    Returns:
        tuple: Tuple of reset gradients, placed in their original order.
    """
    return_grad = [None for _ in in_args]
    for i, grad_i in enumerate(grad_idx):
        return_grad[grad_i] = grad[i]

    return tuple(return_grad) + grad[len(grad_idx):]


class DEQGradCkpt(torch.autograd.Function):
    """
    Implements an autograd Function for maintaining constant memory usage during the backward pass in Deep Equilibrium Models (DEQ).
    """

    @staticmethod
    def _find_params(module):
        """
        Finds parameters in a module that require gradients.

        Args:
            module (torch.nn.Module): Module in which to search for parameters.

        Returns:
            list[torch.Tensor]: List of tuples, each containing the name of a parameter and the corresponding tensor.
        """
        return [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]

    @staticmethod
    def fetch_params(modules):
        """
        Fetches parameters that are tensors and require gradients from a module or a list of modules.

        Args:
            modules (torch.nn.Module or list[torch.nn.Module]): The module(s) to search.

        Returns:
            list[torch.Tensor]: List of tensors corresponding to parameters that require gradients.
        """
        if type(modules) not in [list, tuple]:
            modules = [modules]

        params = []
        for module in modules:
            if getattr(module, "_is_replica", False):
                named_params = module._named_members(get_members_fn=DEQGradCkpt._find_params)
                params += [param for _, param in named_params]
            else:
                params += [param for param in module.parameters() if param.requires_grad]
        
        return params

    @staticmethod
    def forward(
            ctx, func, n_func_args, *args,
            ):
        """
        Runs the forward pass of the given `func` Module.

        Args:
            ctx (object): Context object used for saving intermediate variables that may be needed in the backward pass.
            func (torch.nn.Module): Pytorch Module for which the forward pass and gradients are computed.
            n_func_args (int): Number of input arguments that the function takes.
            *args: Additional arguments.

        Returns:
            tuple[Any]: Output of the `func` Module.
        """
        ctx.func = func
        ctx.in_args, ctx.params = args[:n_func_args], args[n_func_args:]

        out = func(*ctx.in_args)

        return out

    @staticmethod
    def backward(ctx, *out_grad):
        """
        Runs the backward pass by recomputing activations.

        Args:
            ctx (object): Context object that contains intermediate variables from the forward pass.
            *out_grad: Gradients of the output.

        Returns:
            tuple[Any]: Gradients of the input arguments. None for the input arguments that don't require gradient.
        """
        in_args, params = ctx.in_args, ctx.params
        func = ctx.func
        
        forward_args, grad_args, grad_idx = filter_input(in_args)

        with torch.enable_grad():
            out = func(*forward_args)
        out = (out,) if torch.is_tensor(out) else out
        
        out_tensor, out_grad_tensor = filter_out(out, out_grad)

        # Multivariate vjp. 
        grad = torch.autograd.grad(
                out_tensor, grad_args+params,
                out_grad_tensor, 
                retain_graph=True, allow_unused=True
                )
        
        return_grad = reset_grad(grad, in_args, grad_idx)

        return (None, None, *return_grad)


def mem_gc(func, in_args=None):
    """
    Performs the forward and backward pass of a PyTorch Module using gradient checkpointing.

    This function is designed for use with iterative computational graphs and the PyTorch DDP training protocol. 
    In the forward pass, it does not store any activations. 
    During the backward pass, it first recomputes the activations and then applies the vector-Jacobian product (vjp) to calculate gradients with respect to the inputs. 

    The function automatically tracks gradients for the parameters and input tensors that require gradients. 
    It is particularly useful for creating computational graphs with constant memory complexity, i.e., :math:`\\mathcal{O}(1)` memory.

    Args:
        func (torch.nn.Module): Pytorch Module for which gradients will be computed.
        in_args (tuple, optional): Input arguments for the function. Default None.

    Returns:
        tuple: The output of the `func` Module.
    """

    in_args = in_args if in_args else ()
    params = DEQGradCkpt.fetch_params(func)
    return DEQGradCkpt.apply(func, len(in_args), *in_args, *params)

