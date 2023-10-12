import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize, aug_normalized_adjacency
from time import perf_counter

from sklearn import metrics


def params_count(model):
    count = 0 
    for param in model.parameters():
        count = count + np.prod(param.shape)
    
    print('Total params: %f M' % (count / 1000 / 1000))


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True, need_orig=False):
    """
    Load Citation Networks Datasets.
    """
    if 'chains' in dataset_str:
        return load_citation_chain(normalization, cuda, need_orig=need_orig)

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    if need_orig:
        adj_orig = aug_normalized_adjacency(adj, need_orig=True)
        adj_orig = sparse_mx_to_torch_sparse_tensor(adj_orig).float()
        if cuda:
            adj_orig = adj_orig.cuda()

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return [adj, adj_orig] if need_orig else adj, features, labels, idx_train, idx_val, idx_test

def load_citation_chain(normalization, cuda, need_orig=False):
    """load the synthetic dataset: chain"""
    r = np.random.RandomState(42)
    c = 2 # num of classes
    n = 20 # chains for each class
    l = 10 # length of chain
    f = 100 # feature dimension
    tn = 20  # train nodes
    vl = 100 # val nodes
    tt = 200 # test nodes
    noise = 0.00

    chain_adj = sp.coo_matrix((np.ones(l-1), (np.arange(l-1), np.arange(1, l))), shape=(l, l))
    adj = sp.block_diag([chain_adj for _ in range(c*n)]) # square matrix N = c*n*l

    features = r.uniform(-noise, noise, size=(c, n, l, f))
    #features = np.zeros_like(features)
    features[:, :, 0, :c] += np.eye(c).reshape(c, 1, c) # add class info to the first node of chains.
    features = features.reshape(-1, f)

    labels = np.eye(c).reshape(c, 1, 1, c).repeat(n, axis=1).repeat(l, axis=2) # one-hot labels
    labels = labels.reshape(-1, c)

    idx_random = np.arange(c*n*l)
    r.shuffle(idx_random)
    idx_train = idx_random[:tn]
    idx_val = idx_random[tn:tn+vl]
    idx_test = idx_random[tn+vl:tn+vl+tt]

    if need_orig:
        adj_orig = aug_normalized_adjacency(adj, need_orig=True)
        adj_orig = sparse_mx_to_torch_sparse_tensor(adj_orig).float()
        if cuda:
            adj_orig = adj_orig.cuda()

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense() if sp.issparse(features) else features)).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return [adj, adj_orig] if need_orig else adj, features, labels, idx_train, idx_val, idx_test


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def Evaluation(output, labels):
    preds = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    '''
    binary_pred = preds
    binary_pred[binary_pred > 0.0] = 1
    binary_pred[binary_pred <= 0.0] = 0
    '''
    num_correct = 0
    binary_pred = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k = labels[i].sum().astype('int')
        topk_idx = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx] = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    print('total number of correct is: {}'.format(num_correct))
    #print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    #'''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")



def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    tensor = torch.sparse.FloatTensor(indices, values, shape)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def get_spectral_rad(sparse_tensor, tol=1e-5):
    """Compute spectral radius from a tensor"""
    A = sparse_tensor.data.coalesce().cpu()
    A_scipy = sp.coo_matrix((np.abs(A.values().numpy()), A.indices().numpy()), shape=A.shape)
    return np.abs(sp.linalg.eigs(A_scipy, k=1, return_eigenvectors=False)[0]) + tol

def projection_norm_inf(A, kappa=0.99, transpose=False):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v = kappa
    if transpose:
        A_np = A.T.clone().detach().cpu().numpy()
    else:
        A_np = A.clone().detach().cpu().numpy()
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-4)
        # write back
        A_np[idx, :] = a
    A.data.copy_(torch.tensor(A_np.T if transpose else A_np, dtype=A.dtype, device=A.device))
    return A

def projection_norm_inf_and_1(A, kappa_inf=0.99, kappa_1=None, inf_first=True):
    """ project onto ||A||_inf <= kappa return updated A"""
    # TODO: speed up if needed
    v_inf = kappa_inf
    v_1 = kappa_inf if kappa_1 is None else kappa_1
    A_np = A.clone().detach().cpu().numpy()
    if inf_first:
        A_np = projection_inf_np(A_np, v_inf)
        A_np = projection_inf_np(A_np.T, v_1).T
    else:
        A_np = projection_inf_np(A_np.T, v_1).T
        A_np = projection_inf_np(A_np, v_inf)
    A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))
    return A

def projection_inf_np(A_np, v):
    x = np.abs(A_np).sum(axis=-1)
    for idx in np.where(x > v)[0]:
        # read the vector
        a_orig = A_np[idx, :]
        a_sign = np.sign(a_orig)
        a_abs = np.abs(a_orig)
        a = np.sort(a_abs)

        s = np.sum(a) - v
        l = float(len(a))
        for i in range(len(a)):
            # proposal: alpha <= a[i]
            if s / l > a[i]:
                s -= a[i]
                l -= 1
            else:
                break
        alpha = s / l
        a = a_sign * np.maximum(a_abs - alpha, 0)
        # verify
        assert np.isclose(np.abs(a).sum(), v, atol=1e-6)
        # write back
        A_np[idx, :] = a
    return A_np

def clip_gradient(model, clip_norm=10):
    """ clip gradients of each parameter by norm """
    for param in model.parameters():
        torch.nn.utils.clip_grad_norm(param, clip_norm)
    return model

def l_1_penalty(model, alpha=0.1):
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += alpha * torch.sum(torch.abs(param))
    return regularization_loss

class AdditionalLayer(torch.nn.Module):
    def __init__(self, model, num_input, num_output, activation=torch.nn.ReLU()):
        super().__init__()
        self.model = model
        self.add_module("model", self.model)
        self.activation = activation
        if isinstance(activation, torch.nn.Module):
            self.add_module("activation", self.activation)
        self.func = torch.nn.Linear(num_input, num_output, bias=False)

    def forward(self, *input):
        x = self.model(*input)
        x = self.activation(x)
        return self.func(x)

def load_raw_graph(dataset_str = "amazon-all"):
    txt_file = 'data/' + dataset_str + '/adj_list.txt'
    graph = {}
    with open(txt_file, 'r') as f:
        cur_idx = 0
        for row in f:
            row = row.strip().split()
            adjs = []
            for j in range(1, len(row)):
                adjs.append(int(row[j]))
            graph[cur_idx] = adjs
            cur_idx += 1
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    normalization="AugNormAdj"
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj

def load_txt_data(dataset_str = "amazon-all", portion = '0.06'):
    adj = load_raw_graph(dataset_str)
    idx_train = list(np.loadtxt('data/' + dataset_str + '/train_idx-' + str(portion) + '.txt', dtype=int))
    idx_val = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
    idx_test = list(np.loadtxt('data/' + dataset_str + '/test_idx.txt', dtype=int))
    labels = np.loadtxt('data/' + dataset_str + '/label.txt')
    with open('data/' + dataset_str + '/meta.txt', 'r') as f:
        num_nodes, num_class = [int(w) for w in f.readline().strip().split()]

    features = sp.identity(num_nodes)
    
    # porting to pytorch
    features = sparse_mx_to_torch_sparse_tensor(features).float()
    labels = torch.FloatTensor(labels)
    #labels = torch.max(labels, dim=1)[1]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, num_nodes, num_class

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    adj_index = adj.coalesce().indices()
    adj_value = adj.coalesce().values()
    features_index = features.coalesce().indices()
    features_value = features.coalesce().values()
    m = adj.shape[0]
    n = adj.shape[1]
    k = features.shape[1]

    for i in range(degree):
        #features = torch.spmm(adj, features)
        features_index, features_value = torch_sparse.spspmm(adj_index, adj_value, features_index, features_value, m, n, k)
    precompute_time = perf_counter()-t
    return torch.sparse.FloatTensor(features_index, features_value, torch.Size(features.shape)), precompute_time

class SparseDropout(torch.nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob=1-dprob

    def forward(self, x, training):
        if training:
            mask=((torch.rand(x._values().size())+(self.kprob)).floor()).type(torch.bool)
            rc=x._indices()[:,mask]
            val=x._values()[mask]*(1.0/self.kprob)
            return torch.sparse.FloatTensor(rc, val, torch.Size(x.shape))
        else:
            return x



