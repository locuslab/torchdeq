import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from modules.injection import get_filter


class MFNLayer(nn.Module):
    def __init__(self, interm_channels, filter_type='fourier', filter_options={}, bias=True, norm_type='none', init='default'):
        super().__init__()

        self.z_map = nn.Linear(interm_channels, interm_channels, bias=bias)
        nn.init.uniform_(self.z_map.weight, -np.sqrt(1 / interm_channels), np.sqrt(1 / interm_channels))

        self.filter = get_filter(filter_type)

    def forward(self, z, u):
        z = self.z_map(z)
        return self.filter(z, u)


class MFN(nn.Module):
    def __init__(self, interm_channels, n_layers,
            filter_type='fourier', filter_options={},
            norm_type='none', init='default'):
        super().__init__()
        
        # TODO: Test different n_layers and layer_scales for init.
        # orig code uses the same n_layers for all!!!
        self.layers = nn.ModuleList([
            MFNLayer(
                interm_channels=interm_channels, 
                filter_type=filter_type,
                filter_options=filter_options,
                norm_type=norm_type,
                init=init
                ) for _ in range(n_layers)
        ])
        
        if filter_type in ['fourier', 'gabor']:
            init_filter_in = torch.ones(1, interm_channels)
        else:
            init_filter_in = torch.zeros(1, interm_channels)
        self.register_buffer('z_init', init_filter_in)
        
        self.filter = get_filter(filter_type)
    
    def forward(self, z, u):
        # TODO: Test the filter and strange z_init
        if self.filter is not None:
            gx = self.filter(self.z_init, u[0])
            z = z + gx

        for i, layer in enumerate(self.layers):
            z = layer(z, u[i+1])
            
        return z

