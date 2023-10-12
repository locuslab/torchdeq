import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

from modules.layer import MFN
from modules.injection import get_injection


class DEQINR(nn.Module):
    def __init__(
            self, 
            args,
            in_channels, 
            interm_channels, 
            output_channels, 
            input_scale=256., 
            filter_type='fourier', 
            filter_options={},
            norm_type='none',
            n_layers=1, 
            one_pass=False, 
            init='default',
            output_linear=True,
            **kwargs
            ):
        super().__init__()
        
        # TODO: test n_layer and n_layer+1
        self.injection = get_injection(filter_type, 
                in_channels, interm_channels, input_scale, 
                n_layers=n_layers+1, init='default', 
                **filter_options)

        self.f = MFN(interm_channels, n_layers, 
                filter_type=filter_type, norm_type=norm_type, 
                filter_options=filter_options, init=init)
        
        self.output_map = nn.Linear(interm_channels, output_channels)
        
        # Add these for DEQ
        apply_norm(self.f, args=args)

        self.deq = get_deq(args)
        self.args = args

        self.one_pass = one_pass
        self.interm_channels = interm_channels

        if not output_linear:
            raise ValueError('Discarded kwarg output_linear!')
    
    def _decode(self, z_pred):
        return self.output_map(z_pred)

    def forward(self, x, z=None, skip_solver=False, verbose=False, include_grad=False, **kwargs):
        if z is None:
            z = torch.zeros(*x.shape[:-1], self.interm_channels, device=x.device)

        if include_grad:
            x.requires_grad_(True)
        
        u = self.injection(x)

        reset_norm(self.f)
        func = lambda z: self.f(z, u)

        if self.one_pass:
            # Run a one pass model instead of solving equilibrium point
            z_pred = self.f(z, u)
            lowest = 0.
            nstep = 0

            outputs = [self._decode(z_pred)]
        else:
            # DEQ-INR
            solver_kwargs = {'f_max_iter': 0} if skip_solver else {}
            z_out, info = self.deq(func, z, solver_kwargs=solver_kwargs, **kwargs)
            z_pred = z_out[-1]
            lowest = info['rel_lowest']
            nstep = info['nstep'].mean().item()

            outputs = [self._decode(z_pred) for z_pred in z_out]

        ret_dict = {
            'output': outputs[-1],
            'imp_layer_output': z_pred,
            'forward_steps': nstep
        }

        if include_grad:
            grad = torch.autograd.grad(outputs[-1], [x], grad_outputs=torch.ones_like(output), create_graph=True)[0]
            ret_dict['grad'] = grad

        return ret_dict
        

class SIREN(nn.Module):
    def __init__(self, in_channels, interm_channels, out_channels, n_layers=4, scale=100, omega=30., output_linear=True):
        super().__init__()
        _linear_maps = \
            ([nn.Linear(in_channels, interm_channels)]
            + [nn.Linear(interm_channels, interm_channels) for i in range(n_layers - 1)]
            + [nn.Linear(interm_channels, out_channels)])
        
        print(scale)
        self.omega = omega
        self.linear_maps = nn.ModuleList(_linear_maps)
        # self.linear_maps[0].weight.data *= scale
        self.scale = scale
        nn.init.uniform_(self.linear_maps[0].weight, -1 / in_channels, 1 / in_channels)
        for i in range(1, len(self.linear_maps)):
            nn.init.uniform_(
                self.linear_maps[i].weight,
                -np.sqrt(6 / interm_channels) / self.omega,
                np.sqrt(6 / interm_channels) / self.omega
            )
        self.output_linear = output_linear
        self.interm_channels = interm_channels

    def forward(self, x, z=None, **kwargs):
        h = x
        for i in range(len(self.linear_maps) - 1):
            if i > 0:
                h = torch.sin(self.omega * self.linear_maps[i](h))
            else:
                h = torch.sin(self.scale * self.linear_maps[i](h))
        h = self.linear_maps[-1](h)
        if not self.output_linear:
            h = torch.sin(self.omega * h)
        return {
            'output': h
        }


class FFN(nn.Module):
    def __init__(self, in_channels, interm_channels, out_channels, n_layers=4, output_linear=False):
        super().__init__()
        self.embedder = GaussianFFNEmbedder(in_channels)
        _linear_maps = \
            ([nn.Linear(self.embedder.out_channels, interm_channels)]
            + [nn.Linear(interm_channels, interm_channels) for i in range(n_layers - 1)]
            + [nn.Linear(interm_channels, out_channels)])
        
        self.linear_maps = nn.ModuleList(_linear_maps)
        self.output_linear = output_linear
        self.interm_channels = interm_channels
    
    def forward(self, x, z=None, **kwargs):
        h = self.embedder(x)
        for m in self.linear_maps[:-1]:
            h = torch.relu(m(h))
        if not self.output_linear:
            h = torch.sin(self.linear_maps[-1](h))
        return {
            'output': h
        }


