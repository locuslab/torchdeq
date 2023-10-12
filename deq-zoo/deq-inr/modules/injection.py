import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules.utils import sql2_dist


class FourierInjection(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1., n_layers=1):
        super().__init__()
        
        interm_channels = interm_channels * n_layers
        self.n_layers = n_layers

        self.x_map = nn.Linear(in_channels, interm_channels)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        out = torch.sin(self.x_map(x))
        return out.chunk(self.n_layers, dim=-1)


class FourierFilter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z, u):
        return u * z


class GaborInjection(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1., alpha=5., beta=1., n_layers=1):
        super().__init__()

        interm_channels = interm_channels * n_layers
        self.n_layers = n_layers

        self.x_map = nn.Linear(in_channels, interm_channels)

        self.mu = nn.Parameter(
                torch.rand(interm_channels, in_channels) * 2 - 1
                )
        self.gamma = nn.Parameter(
                torch.distributions.gamma.Gamma(alpha, beta).sample((interm_channels,))
                )

        self.x_map.weight.data *= scale * torch.sqrt(self.gamma[:, None])
        self.x_map.bias.data.uniform_(-np.pi, np.pi)
  
    def forward(self, x):
        periodic = torch.sin(self.x_map(x))
        local = torch.exp(-0.5 * self.gamma[None, :] * sql2_dist(x.squeeze(0), self.mu))
        scale = periodic * local

        return scale.chunk(self.n_layers, dim=-1)


class GaborFilter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z, u):
        return u * z


class SIRENInjection(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=1., n_layers=1):
        super().__init__()
        
        interm_channels = interm_channels * n_layers
        self.n_layers = n_layers

        self.x_map = nn.Linear(in_channels, interm_channels)
        self.x_map.weight.data *= scale
        self.x_map.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        out = self.x_map(x)
        return out.chunk(self.n_layers, dim=-1)


class SIRENFilter(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z, u):
        return torch.sin(z + u)


class GaussianFFNEmbedder(nn.Module):
    def __init__(self, in_channels, embedding_size=256, scale=10.):
        super().__init__()

        self.bval = nn.Parameter(torch.randn(2, embedding_size) * 10, requires_grad=False)
        self.aval = nn.Parameter(torch.ones(embedding_size), requires_grad=False)

        self.out_channels = embedding_size * 2
    
    def forward(self, x):
        return torch.cat([self.aval * torch.sin((2 * np.pi * x) @ self.bval),
                          self.aval * torch.cos((2 * np.pi * x) @ self.bval)], dim=1)


class FFNInjection(nn.Module):
    def __init__(self, in_channels, interm_channels, scale=10.):
        super().__init__()
        
        interm_channels = interm_channels * n_layers
        self.n_layers = n_layers

        self.embedder = GaussianFFNEmbedder(in_channels, interm_channels // 2, scale=scale)
        self.x_map = nn.Linear(self.embedder.out_channels, interm_channels)
    
    def forward(self, x):
        out = self.x_map(self.embedder(x))
        return out.chunk(self.n_layers, dim=-1)


class FFNFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedder = GaussianFFNEmbedder(in_channels, interm_channels // 2, scale=scale)
        self.x_map = nn.Linear(self.embedder.out_channels, interm_channels)
    
    def forward(self, z, u):
        return torch.relu(z + u)


def get_filter(filter_type):
    if filter_type == 'fourier':
        return FourierFilter()
    elif filter_type == 'gabor':
        return GaborFilter()
    elif filter_type == 'siren_like':
        return SIRENFilter()
    elif filter_type == 'relu':
        return FFNFilter()
    else:
        raise ValueError("Filter {:s} not defined".format(filter_type))


def get_injection(filter_type, in_channels, interm_channels, scale, n_layers=1, init='default', **filter_options):
    # n_layers = n_layers + 1
    layer_scale = np.sqrt(n_layers)
    # layer_scale = np.sqrt(2 * n_layers)
    # layer_scale = 0.5 * np.sqrt(n_layers)

    if filter_type == 'fourier':
        return FourierInjection(in_channels, interm_channels, scale / layer_scale, n_layers=n_layers)
    elif filter_type == 'gabor':
        if 'alpha' in filter_options:
            # alpha = filter_options['alpha'] / n_layers
            alpha = float(filter_options['alpha'])
        else:
            alpha = 6.0
        return GaborInjection(in_channels, interm_channels, scale / layer_scale, alpha, n_layers=n_layers)
    elif filter_type == 'siren_like':
        return SIRENInjection(in_channels, interm_channels, scale / layer_scale, n_layers=n_layers)
    elif filter_type == 'relu':
        return FFNInjection(in_channels, interm_channels, 15., n_layers=n_layers)
    else:
        raise ValueError("Injection {:s} not defined".format(filter_type))


