import math
import numpy as np

import torch
import torch.sparse
import torch.nn as nn
import torch.nn.functional as F

from torchdeq import get_deq
from torchdeq.norm import register_norm_module, apply_norm, reset_norm

from utils import projection_norm_inf, projection_norm_inf_and_1, SparseDropout


class ImplicitGraph(nn.Module):
    """
    A Implicit Graph Neural Network Layer (IGNN)
    """

    def __init__(self, args, in_features, out_features, num_node, kappa=0.99, b_direct=False):
        super(ImplicitGraph, self).__init__()
        self.p = in_features
        self.m = out_features
        self.n = num_node
        self.k = kappa      # if set kappa=0, projection will be disabled at forward feeding.
        self.b_direct = b_direct

        self.W = nn.Parameter(torch.FloatTensor(self.m, self.m))
        self.Omega_1 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.Omega_2 = nn.Parameter(torch.FloatTensor(self.m, self.p))
        self.bias = nn.Parameter(torch.FloatTensor(self.m, 1))
        self._init()

        apply_norm(self, args=args)
        self.deq = get_deq(args)

    def _init(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.Omega_1.data.uniform_(-stdv, stdv)
        self.Omega_2.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, X_0, A, U, phi, A_rho=1.0, fw_mitr=300, bw_mitr=300, A_orig=None):
        """Allow one to use a different A matrix for convolution operation in equilibrium equ"""
        reset_norm(self)
        if self.k is not None: # when self.k = 0, A_rho is not required
            self.W = projection_norm_inf(self.W, kappa=self.k/A_rho)

        support_1 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_1.T).T
        support_1 = torch.spmm(torch.transpose(A, 0, 1), support_1.T).T
        support_2 = torch.spmm(torch.transpose(U, 0, 1), self.Omega_2.T).T
        b_Omega = support_1 + support_2
        
        # Define variables
        W, A, B = self.W, A if A_orig is None else A_orig, b_Omega
        X = B if X_0 is None else X_0
        At = torch.transpose(A, 0, 1)
        
        def func(X_now):
            X_ = W @ X_now
            support = torch.spmm(At, X_.T).T
            X_next = phi(support + B)
            return X_next
        
        X_out, info = self.deq(func, X)
        return X_out[-1]


# Register self-defined layer for Norms 
register_norm_module(ImplicitGraph, 'weight_norm', names=['W', 'Omega_1', 'Omega_2'], dims=[0, 0, 0])
register_norm_module(ImplicitGraph, 'spectral_norm', names=['W', 'Omega_1', 'Omega_2'], dims=[0, 0, 0])


