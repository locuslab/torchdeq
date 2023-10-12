import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import ImplicitGraph
from torch.nn import Parameter
from utils import get_spectral_rad, SparseDropout
import torch.sparse as sparse
from torch_geometric.nn import global_add_pool


class IGNN(nn.Module):
    def __init__(self, args, nfeat, nhid, nclass, num_node, dropout, kappa=0.9, adj_orig=None):
        super(IGNN, self).__init__()

        self.adj = None
        self.adj_rho = None
        self.adj_orig = adj_orig

        #three layers and two MLP
        self.ig1 = ImplicitGraph(args, nfeat, nhid, num_node, kappa)
        self.ig2 = ImplicitGraph(args, nhid, nhid, num_node, kappa)
        self.ig3 = ImplicitGraph(args, nhid, nhid, num_node, kappa)
        self.dropout = dropout
        self.X_0 = None
        self.V_0 = nn.Linear(nhid, nhid)
        self.V_1 = nn.Linear(nhid, nclass)

    def forward(self, features, adj, batch):
        '''
        if adj is not self.adj:
            self.adj = adj
            self.adj_rho = get_spectral_rad(adj)
        '''
        self.adj_rho = 1

        x = features

        #three layers and two MLP
        x = self.ig1(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        x = self.ig2(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig)
        x = self.ig3(self.X_0, adj, x, F.relu, self.adj_rho, A_orig=self.adj_orig).T
        x = global_add_pool(x, batch)
        x = F.relu(self.V_0(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.V_1(x)
        return F.log_softmax(x, dim=1)


