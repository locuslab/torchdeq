from __future__ import division
from __future__ import print_function

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchdeq.utils import add_deq_args

from utils import accuracy, clip_gradient, load_citation, Evaluation, AdditionalLayer
from models_chains import IGNN

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0.9,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)')
parser.add_argument('--dataset', type=str, default="chains-10",
                        help='Dataset to use.')
parser.add_argument('--feature', type=str, default="mul",
                    choices=['mul', 'cat', 'adj'],
                    help='feature-type')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['AugNormAdj'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--per', type=int, default=-1,
                    help='Number of each nodes so as to balance.')
parser.add_argument('--experiment', type=str, default="base-experiment",
                    help='feature-type')

# DEQ techniques
add_deq_args(parser)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.dataset, args.normalization, args.cuda)


# Model and optimizer
model = IGNN(args,
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            num_node = adj.shape[1],
            dropout=args.dropout,
            kappa=args.kappa)

params_count(model)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()#[:10]
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features = features.T

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    output = F.log_softmax(output, dim=1)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    clip_gradient(model, clip_norm=0.5)
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
        output = F.log_softmax(output, dim=1)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'loss_test: {:.4f}'.format(loss_test.item()),
          'acc_test: {:.4f}'.format(acc_test.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Dataset: " + args.dataset)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
