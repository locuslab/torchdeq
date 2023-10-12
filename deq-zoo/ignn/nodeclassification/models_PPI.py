import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.sparse as sparse

from layers import ImplicitGraph
from utils import get_spectral_rad, SparseDropout


class IGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()
        print(args)

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig
        
        self.z_0 = None

        self.layers = 5
        self.ignn = nn.ModuleList([
            ImplicitGraph(args, nfeat, 4*nhid, num_node, kappa),
            ImplicitGraph(args, 4*nhid, 2* nhid, num_node, kappa),
            ImplicitGraph(args, 2*nhid, 2*nhid, num_node, kappa),
            ImplicitGraph(args, 2*nhid, nhid, num_node, kappa),
            ImplicitGraph(args, nhid, nclass, num_node, kappa)
            ])
        
        self.proj = nn.ModuleList([
            nn.Linear(nfeat, 4*nhid),
            nn.Linear(4*nhid, 2*nhid),
            nn.Linear(2*nhid, 2*nhid),
            nn.Linear(2*nhid, nhid),
            nn.Linear(nhid, nclass)
            ])
        self.act = nn.ELU()

    def forward(self, features, adj):
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)

        x = features
        for i in range(self.layers):
            z_star = self.ignn[i](
                    self.z_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig
                    )
            x = z_star + self.proj[i](x.T).T
            if i+1 < self.layers:
                x = self.act(x)
        
        return x.T


