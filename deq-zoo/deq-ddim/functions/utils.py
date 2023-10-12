import torch
import numpy as np
from scipy.stats import ortho_group

@torch.no_grad()
def get_ortho_mat(mat_type, batched_image_shape, device, method='bases'):
    edge_dim = batched_image_shape[1] * batched_image_shape[2] * batched_image_shape[3]
    if method == 'scipyv2':
        U = ortho_group.rvs(dim=batched_image_shape[2])
        U = torch.from_numpy(U).type(mat_type).to(device)
    elif method == 'bases':
        col_idx = torch.randperm(edge_dim)
        row_idx = torch.arange(edge_dim)
        U = torch.zeros(size=(edge_dim, edge_dim))
        ones = torch.ones(edge_dim)
        ones[torch.rand(edge_dim) > 0.5] *= -1
        U[row_idx, col_idx] = ones
        U = U.to(device)
    elif method == 'std-bases':
        col_idx = torch.randperm(edge_dim)
        row_idx = torch.arange(edge_dim)
        U = torch.zeros(size=(edge_dim, edge_dim))
        ones = torch.ones(edge_dim)
        U[row_idx, col_idx] = ones
        U = U.to(device)
    elif method == 'normal':
        U = torch.randn((edge_dim, edge_dim), device=device)
        U /= np.sqrt(edge_dim)
    elif method == 'hadamard':
        U = torch.from_numpy(get_hadamard_matrix(edge_dim)).type(mat_type).to(device)
        U /= np.sqrt(edge_dim)
    elif method == 'qr':
        U = torch.randn((edge_dim, edge_dim), device=device)
        U, _ = torch.qr(U)
    return U

def get_identity3D(shape):
    I = torch.eye(shape)
    return torch.cat([I, I, I], dim=0)

def combine_tensor(x, y, sigma=None):
    B = x.shape[0]
    if sigma is None:
        return torch.cat([x.view(B, -1), y.view(B, -1)], dim=1)
    return torch.cat([x.view(B, -1), y.view(B, -1), sigma.view(B, 1)], dim=1)

def split_combined_tensor(x, image_dims, reshape=True):
    dims = x.shape[1]
    B, C, H, W = image_dims
    if dims == 2*C*H*W + 1:
        split_tensor = torch.split(x, [C*H*W, C*H*W, 1], dim=1)
        if reshape:
            x_mod_t = split_tensor[0].view(image_dims)
            noise_t = split_tensor[1].view(image_dims)
            sigma_t = split_tensor[2]
        else:
            x_mod_t = split_tensor[0]
            noise_t = split_tensor[1]
            sigma_t = split_tensor[2]
        return x_mod_t, noise_t, sigma_t
    elif dims == 2*C*H*W:
        return split_combined_tensor2(x, image_dims, reshape)

def split_combined_tensor2(x, image_dims, reshape=True):
    B, C, H, W = image_dims
    split_tensor = torch.split(x, [C*H*W, C*H*W], dim=1)

    if reshape:
        x_mod_t = split_tensor[0].view(image_dims)
        noise_t = split_tensor[1].view(image_dims)
    else:
        x_mod_t = split_tensor[0]
        noise_t = split_tensor[1]
    return x_mod_t, noise_t

def get_hadamard_matrix(dim=64*64*3):
    H_12 = [
    [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1],
    [1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1],
    [1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1],
    [1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  1,  1],
    [1,  1,  1, -1,  1,  1, -1,  1, -1, -1, -1,  1],
    [1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1, -1],
    [1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1, -1],
    [1, -1, -1,  1,  1,  1, -1,  1,  1, -1,  1, -1],
    [1, -1, -1, -1,  1,  1,  1, -1,  1,  1, -1,  1],
    [1,  1, -1, -1, -1,  1,  1,  1, -1,  1,  1, -1],
    [1, -1,  1, -1, -1, -1,  1,  1,  1, -1,  1,  1],
    ]

    H_12 = np.array(H_12)
    H_2 = [
        [1, 1],
        [1, -1]
    ]
    H_2 = np.array(H_2)

    H = np.kron(H_2, H_12)
    while H.shape[0] < dim:
        H = np.kron(H_2, H)

    assert H.shape[0] == dim, "dimensions of matrix doesn't match the expected dimension"
    return H
    