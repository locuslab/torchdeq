from logging import log
from turtle import pd

import torch
from torch import nn

import wandb
import numpy as np
import torch.autograd as autograd

from torchdeq import get_deq


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


class DEQLatentSpaceOpt(nn.Module):
    def __init__(self, args, model, sddim=False):
        super().__init__()

        self.deq = get_deq(args)
        self.args = args

        self.model = model
        self.sddim = sddim
    
    def get_ddim_injection(
            self, all_xt, seq, betas, batch_size, 
            eta=0, all_noiset=None):
        cur_seq = list(seq)
        seq_next = [-1] + list(seq[:-1])

        gather_idx = [idx for idx in range(len(cur_seq), len(all_xt), len(cur_seq)+2)]
        xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq)+1)]
        next_idx = [idx for idx in range(len(all_xt)) if idx not in gather_idx]
        prev_idx = [idx + 1 for idx in next_idx]

        T = len(cur_seq)
        t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)
        next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())
        alpha_ratio = (at_next/at[0]).sqrt() 
        
        if self.sddim:
            sigma_t = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            noise_t = (1 / at_next.sqrt()) * sigma_t * all_noiset
            et_coeff2 = (1 - at_next - sigma_t**2).sqrt() - (((1 - at)*at_next)/at).sqrt()
        else:
            noise_t = None
            et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()
        
        et_coeff = (1 / at_next.sqrt()) * et_coeff2
        et_prevsum_coeff = at_next.sqrt()

        diffusion_args = {
            "T" : T, 
            "t" : t,
            "xT_idx": xT_idx,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            'at': at,
            'at_next': at_next,
            "alpha_ratio": alpha_ratio,
            'et_coeff': et_coeff, 
            'et_prevsum_coeff': et_prevsum_coeff,
            'noise_t': noise_t
        }

        return diffusion_args

    # This method assumes that a single image is being inverted!
    def joint_diffusion(
            self, xt, t, xT, all_xT, 
            next_idx, xT_idx, prev_idx,
            et_coeff, et_prevsum_coeff, 
            noise_t=None,
            **kwargs):
        xt_in = xt[next_idx]
        et = self.model(xt_in, t)
        
        if self.sddim:
            et_updated = et_coeff * et + noise_t # Additional noise
        else:
            et_updated = et_coeff * et

        et_prevsum = et_updated.cumsum(dim=0)
        xt_next = all_xT + et_prevsum_coeff * et_prevsum

        xt_all = torch.zeros_like(xt)
        xt_all[xT_idx] = xT
        xt_all[prev_idx] = xt_next
        xt_all = xt_all.cuda()

        return xt_all

    def forward(self, x, injection, anderson_params=None, logger=None):        
        injection['xT'] = x[0].unsqueeze(0)
        injection['all_xT'] = injection['alpha_ratio'] * \
                torch.repeat_interleave(injection['xT'], injection['T'], dim=0).to(x.device)

        if anderson_params is None:
            anderson_params = {
                "m": 5,
                "lam": 1e-3,
                "beta": 1.0
            }

        def func(x):
            return self.joint_diffusion(x, **injection)

        x_pred, info = self.deq(func, x, solver_kwargs=anderson_params)
        x_eq = x_pred[-1]

        if logger is not None:
            logger({"generated images": [wandb.Image(x_eq[i].view((3, 32, 32))) for i in list(range(0, T, T//10)) + [-5, -4, -3, -2, -1]]})
        
        return x_eq


