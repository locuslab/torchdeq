from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import time
import argparse
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool

from torchdeq.utils import add_deq_args

from utils import params_count, load_citation, accuracy, clip_gradient, l_1_penalty, get_spectral_rad
from models import IGNN

from normalization import fetch_normalization


# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0.9,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)')
parser.add_argument('--dataset', type=str, default="PROTEINS",
                        help='Dataset to use.')
parser.add_argument('--fold_idx', type=int, default=0,
                    help='Which fold is chosen for test (0-9).')

# DEQ techniques
add_deq_args(parser)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
dataset = TUDataset(path, name=args.dataset).shuffle()
skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = args.seed)
idx_list = []
for idx in skf.split(np.zeros(len(dataset.data.y)), dataset.data.y):
    idx_list.append(idx)
assert 0 <= args.fold_idx and args.fold_idx < 10, "fold_idx must be from 0 to 9."

# Model and optimizer
device = torch.device('cuda' if args.cuda else 'cpu')

results = [[] for i in range(10)]

for fold_idx in range(10):
    # if dataset.num_features == 0:
    #     dataset.num_features = 1
    model = IGNN(args,
                nfeat=dataset.num_features,
                nhid=args.hidden,
                nclass=dataset.num_classes,
                num_node=None,
                dropout=args.dropout,
                kappa=args.kappa).cuda()

    params_count(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    train_idx, test_idx = idx_list[fold_idx]
    test_dataset = dataset[test_idx.tolist()]
    train_dataset = dataset[train_idx.tolist()]

    test_loader = DataLoader(test_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)

    def train(epoch):
        model.train()

        if epoch == 51:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr']

        loss_all = 0
        for data in train_loader:
            data = data.cuda()
            optimizer.zero_grad()
            if data.edge_attr is None:
                edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)
            else:
                edge_weight = data.edge_attr
            adj_sp = csr_matrix((edge_weight.cpu().numpy(), (data.edge_index[0,:].cpu().numpy(), data.edge_index[1,:].cpu().numpy() )), shape=(data.num_nodes, data.num_nodes))
            adj_normalizer = fetch_normalization("AugNormAdj")
            adj_sp_nz = adj_normalizer(adj_sp)
            adj = torch.sparse.FloatTensor(torch.LongTensor(np.array([adj_sp_nz.row,adj_sp_nz.col])).cuda(), torch.Tensor(adj_sp_nz.data).cuda(), torch.Size([data.num_nodes, data.num_nodes])) #normalized adj

            adj_ori = torch.sparse.FloatTensor(data.edge_index, edge_weight, torch.Size([data.num_nodes, data.num_nodes])) #original adj
            if data.x is None:
                data.x = torch.sparse.sum(adj_ori, [0]).to_dense().unsqueeze(1).cuda()
            output = model(data.x.T, adj, data.batch)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return loss_all / len(train_dataset)


    def test(loader):
        model.eval()

        correct = 0
        for data in loader:
            data = data.cuda()
            if data.edge_attr is None:
                edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)
            else:
                edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)
            adj_sp = csr_matrix((edge_weight.cpu().numpy(), (data.edge_index[0,:].cpu().numpy(), data.edge_index[1,:].cpu().numpy() )), shape=(data.num_nodes, data.num_nodes))
            adj_sp = adj_sp + adj_sp.T
            adj_normalizer = fetch_normalization("AugNormAdj")
            adj_sp_nz = adj_normalizer(adj_sp)
            adj = torch.sparse.FloatTensor(torch.LongTensor(np.array([adj_sp_nz.row,adj_sp_nz.col])).cuda(), torch.Tensor(adj_sp_nz.data).cuda(), torch.Size([data.num_nodes, data.num_nodes])) #normalized adj

            adj_ori = torch.sparse.FloatTensor(data.edge_index, edge_weight, torch.Size([data.num_nodes, data.num_nodes])) #original adj

            if data.x is None:
                data.x = torch.sparse.sum(adj_ori, [0]).to_dense().unsqueeze(1).cuda()
            output = model(data.x.T, adj, data.batch)
            pred = output.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)


    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        results[fold_idx].append(test_acc)
        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                           train_acc, test_acc))
re_np = np.array(results)
re_all = [re_np.max(1).mean(), re_np.max(1).std()]
print('Graph classification mean accuracy and std are {}'.format(re_all))
