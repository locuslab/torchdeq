import torch
import torch.nn as nn

import time
import numpy as np
from tqdm import tqdm

import plyfile
import skimage.measure
import re
import configargparse

def network_spec(s):
    if type(s) != str:
        return s
    return list(map(int, s.strip('[] ').split(',')))

def get_psnr(output, target):
        mse = np.mean((output - target) ** 2)
        return -10 * np.log10(mse)

def get_psnr_stats(output, target):
    psnr_list = []
    for i in range(len(output)):
        psnr_list.append(get_psnr(output[i], target[i]))
    
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)

    return psnr_list, psnr_mean, psnr_std

class ConfusionMatrix:
    def __init__(self):
        self.reset()
    
    def update(self, pred, label):
        assert len(pred.shape) == len(label.shape) and [pred.shape[i] == label.shape[i] for i in range(len(pred.shape))]
        pos_mask = pred == 1
        neg_mask = pred == 0
        true_mask = pred == label

        self.stats['TP'] += np.sum(pos_mask & true_mask)
        self.stats['TN'] += np.sum(neg_mask & true_mask)
        self.stats['FP'] += np.sum(pos_mask & (~true_mask))
        self.stats['FN'] += np.sum(neg_mask & (~true_mask))
    
    def reset(self):
        self.stats = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0
        }
    
    def get_acc(self):
        return (self.stats['TP'] + self.stats['TN']) / (self.stats['TP'] + self.stats['TN'] + self.stats['FP'] + self.stats['FN'])
    
    def get_precision(self):
        return (self.stats['TP']) / (self.stats['TP'] + self.stats['FP'])
    
    def get_recall(self):
        return (self.stats['TP']) / (self.stats['TP'] + self.stats['FN'])
    
    def get_iou(self):
        return (self.stats['TP']) / (self.stats['TP'] + self.stats['FN'] + self.stats['FP'])


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        b = x.size(0)
        return x.view(b, -1)


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
    
    def forward(self, x):
        return x.contiguous().view(-1, *self.target_shape)


class Debug(nn.Module):
    def __init__(self):
        super(Debug, self).__init__()
    
    def forward(self, x):
        print(x.shape)
        return x


def sql2_dist(x, y):
    # assumes x, y to be 2d matrices (n, d), (k, d) -> (n, k)
    X_sq = x.square().sum(1).unsqueeze(1)
    Y_sq = y.square().sum(1).unsqueeze(1)

    XYT = torch.matmul(x, y.T)
    dist = X_sq + Y_sq.T - 2 * XYT
    return dist


def batched_apply(f, inputs, batch_size=1024, f_kwargs={}, print_progress=False, device=None):
    if type(inputs) not in [list, tuple]:
        inputs = [inputs]

    outputs = []
    _iter = range(int(np.ceil(len(inputs[0]) / batch_size)))
    if print_progress:
        _iter = tqdm(_iter)
    for i in _iter:
        batch_out = f(*[t[i * batch_size: (i + 1) * batch_size].to(device) 
        if device is not None else t[i * batch_size: (i + 1) * batch_size] for t in inputs], **f_kwargs)['output']
        outputs.append(batch_out)
    return torch.cat(outputs, dim=0)


def batch_indices_generator(full_size, batch_size, shuffle=False, max_rounds=-1):
    round_cnt = 0
    while max_rounds == -1 or round_cnt >= max_rounds:
        indices = np.random.permutation(full_size) if shuffle else np.arange(full_size)
        for i in range(int(np.ceil(full_size / batch_size))):
            yield indices[i * batch_size: (i + 1) * batch_size]
        round_cnt += 1


def create_mesh(
    model, filename, N=256, max_batch=512 ** 2, offset=None, scale=None, device='cuda'
):
    start = time.time()
    ply_filename = filename

    model.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].to(device)

        samples[head : min(head + max_batch, num_samples), 3] = (
            model(sample_subset)['output']
            .squeeze()#.squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

