import torch


def mixed_init(z_shape, device=None):
    """
    Initializes a tensor with a shape of `z_shape` with half Gaussian random values and hald zeros.

    Proposed in the paper, `Path Independent Equilibrium Models Can Better Exploit Test-Time Computation <https://arxiv.org/abs/2211.09961>`_,
    for better path independence.
    
    Args:
        z_shape (tuple): Shape of the tensor to be initialized.
        device (torch.device, optional): The desired device of returned tensor. Default None.

    Returns:
        torch.Tensor: A tensor of shape `z_shape` with values randomly initialized and zero masked.
    """
    z_init = torch.randn(*z_shape, device=device)
    mask = torch.zeros_like(z_init, device=device).bernoulli_(0.5)
    
    return z_init * mask

