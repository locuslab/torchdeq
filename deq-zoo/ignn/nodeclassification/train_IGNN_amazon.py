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

from utils import accuracy, clip_gradient, load_txt_data, Evaluation, AdditionalLayer
from models_amazon import IGNN

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=2333, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0.95,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)')
parser.add_argument('--dataset', type=str, default="amazon-all",
                        help='Dataset to use.')
parser.add_argument('--feature', type=str, default="mul",
                    choices=['mul', 'cat', 'adj'],
                    help='feature-type')
parser.add_argument('--normalization', type=str, default='AugNormAdj',
                   choices=['AugNormAdj'],
                   help='Normalization method for the adjacency matrix.')
parser.add_argument('--degree', type=int, default=2,
                    help='degree of the approximation.')
parser.add_argument('--portion', type=float, default=0.06,
                    help='training set fraction for amazon dataset.')
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
if args.dataset == "amazon-all" or args.dataset == "amazon-top5000":
    portion = args.portion
    adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class = load_txt_data(args.dataset, portion)
else:
    print("dataset provided is not supported")

# Model and optimizer
model = IGNN(args,
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=num_class,
            num_node=num_nodes,
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
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

criterion = nn.BCEWithLogitsLoss()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    f1_train_micro, f1_train_macro = Evaluation(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = criterion(output[idx_val], labels[idx_val])
    #f1_val_micro, f1_val_macro = Evaluation(output[idx_test], labels[idx_test])
    loss_test = criterion(output[idx_test], labels[idx_test])
    f1_test_micro, f1_test_macro = Evaluation(output[idx_test], labels[idx_test])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          "f1_train_micro= {:.4f}".format(f1_train_micro),
          "f1_train_macro= {:.4f}".format(f1_train_macro),
          #'loss_val: {:.4f}'.format(loss_val.item()),
          #"f1_val_micro= {:.4f}".format(f1_val_micro),
          #"f1_val_micro= {:.4f}".format(f1_val_macro),
          'loss_test: {:.4f}'.format(loss_test.item()),
          "f1_test_micro= {:.4f}".format(f1_test_micro),
          "f1_test_macro= {:.4f}".format(f1_test_macro),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = criterion(output[idx_test], labels[idx_test])
    f1_test_micro, f1_test_macro = Evaluation(output[idx_test], labels[idx_test])
    print("Dataset: " + args.dataset)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1_test_micro= {:.4f}".format(f1_test_micro),
          "f1_test_macro= {:.4f}".format(f1_test_macro))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
