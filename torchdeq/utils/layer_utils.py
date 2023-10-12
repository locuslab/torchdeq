import numpy as np

import torch


class DEQWrapper:
    def __init__(self, func, z_init):
        self.func = func
    
    def list2vec(self, z):
        '''
        Converts list of tensors to a batched vector (B, ...)
        '''
        return z

    def vec2list(self, z):
        '''
        Converts a batched vector back to a list
        '''
        return z

    def detach(self, z):
        '''
        Detach gradient from the input tensor
        '''
        return z.detach()

    def __call__(self, z, tau=1.0):
        '''
        A function call to the DEQ f
        '''
        z_out = self.func(z)
        if tau == 1.0:
            return z_out
        else:
            return tau * z_out + (1 - tau) * z
    
    def norm_diff(self, z_new, z_old, **kwargs):
        return (z_new - z_old).norm(p=1)


class MDEQWrapper:
    def __init__(self, func, z_init):
        z_shape = []
        z_indexing = [0]
        for z in z_init:
            z_shape.append(z.shape)
            z = z.flatten(start_dim=1) if z.dim() >= 2 else z.view(z.nelement(), 1)
            z_indexing.append(z[0].nelement())
        
        self.func = func
        self.z_shape = z_shape
        self.z_indexing = np.cumsum(z_indexing)
    
    def list2vec(self, z_list):
        '''
        Converts a list of tensors to a batched vector (B, ...)
        '''
        z_list = [z.flatten(start_dim=1) if z.dim() >= 2 else z.view(z.nelement(), 1) for z in z_list]
        return torch.cat(z_list, dim=1)

    def vec2list(self, z_hid):
        '''
        Converts a batched vector back to a tensor list
        '''
        z_list = []
        z_indexing = self.z_indexing 
        for i, shape in enumerate(self.z_shape):
            z_list.append(z_hid[:, z_indexing[i]:z_indexing[i+1]].view(shape))
        return z_list

    def detach(self, z_hid):
        '''
        Detach gradient from the input tensor
        '''
        return z_hid.detach()

    def __call__(self, z_hid, tau=1.0):
        '''
        A function call to the DEQ f
        '''
        z_list = self.vec2list(z_hid)
        z_list = self.func(*z_list)
        z_out_hid = self.list2vec(z_list)
        
        if tau == 1.0:
            return z_out_hid
        else:
            return tau * z_out_hid + (1 - tau) * z_hid
    
    def norm_diff(self, z_new, z_old, show_list=False):
        if show_list:
            z_new, z_old = self.vec2list(z_new), self.vec2list(z_old)
            return [(z_new[i] - z_old[i]).norm(p=1).item() for i in range(len(z_new))]
        
        return (z_new - z_old).norm(p=1).item()


class SpeedyMDEQWrapper(DEQWrapper):
    def __init__(self, func, z_init):
        super().__init__(func, z_init)

    def detach(self, z_list):
        '''
        Detach gradients from all tensors in the input
        '''
        return [each.detach() for each in z_list]

    def __call__(self, z_list, tau=1.0):
        '''
        A function call to the DEQ f
        '''
        z_out = self.func(*z_list)
        
        if tau == 1.0:
            return z_out
        else:
            return [tau * z_new + (1 - tau) * z for z_new, z in zip(z_out, z_list)]

    def norm_diff(self, z_new, z_old, show_list=False):
        diff = [(z_new[i] - z_old[i]).norm(p=1).item() for i in range(len(z_new))]
        
        if show_list:
            return diff
        else:
            return np.sum(diff)


def deq_decorator(func, z_init=None, no_stat=True):
    if torch.is_tensor(z_init):
        return DEQWrapper(func, z_init), z_init
    else:
        assert type(z_init) in (tuple, list)
        
        if no_stat:
            return SpeedyMDEQWrapper(func, z_init), z_init
        else:
            func = MDEQWrapper(func, z_init)
            return func, func.list2vec(z_init)

