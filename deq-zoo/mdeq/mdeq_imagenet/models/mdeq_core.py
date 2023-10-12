from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

# TorchDEQ
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

BN_MOMENTUM = 0.1
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, DEQ_EXPAND*planes, stride)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, DEQ_EXPAND*planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(DEQ_EXPAND*planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        
        self.downsample = downsample
        self.stride = stride
        
        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.conv2(out) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))

        return out
    
       
blocks_dict = {
    'BASIC': BasicBlock
}


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y
    
    
class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        conv3x3s = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=True))]
            if k != (level_diff-1):
                components.append(('relu', nn.ReLU(inplace=True)))
            conv3x3s.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*conv3x3s)  

    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()

        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res
        
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=True)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))
                   ]))
       
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()

        self.num_branches = num_branches
        self.num_channels = num_channels
        
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=True))
            ])) for i in range(num_branches)])
        self.relu = nn.ReLU(False)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer
        """
        branch_layers = [
                self._make_one_branch(i, block, num_blocks, num_channels) 
                for i in range(num_branches)
                ]
        
        # branch_layers[i] gives the module that operates on input from resolution i
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))
        
        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
            
        return x_fuse


class MDEQNet(nn.Module):

    def __init__(self, args, bn_momentum=0.1):
        """
        Build an MDEQ model with the given hyperparameters
        """
        super(MDEQNet, self).__init__()

        self.num_branches = 4
        self.num_blocks = [1, 1, 1, 1]
        self.num_channels = [32, 64, 128, 256]
        self.init_chansize = self.num_channels[0]
        self.num_layers = 5
        self.num_classes = 1000
        self.downsample_times = 2

        global BN_MOMENTUM, DEQ_EXPAND, NUM_GROUPS
        BN_MOMENTUM = bn_momentum
        DEQ_EXPAND = 5
        NUM_GROUPS = 8
        
        self.f_max_iter = 26
        self.b_max_iter = 27
        
        self.args = args

        self.downsample = nn.Sequential(
            conv3x3(3, self.init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(),
            conv3x3(self.init_chansize, self.init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU()
        )

        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(
                    nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
                    nn.ReLU()
            )

        # PART II: MDEQ's f_\theta layer
        self.full_stage = MDEQModule(self.num_branches, BasicBlock, self.num_blocks, self.num_channels)

        # PART III: Call TorchDEQ.
        apply_norm(self.full_stage, args=args, filter_out='fuse_layers')
        self.deq = get_deq(args)

    def _forward(self, x, **kwargs):
        x = self.downsample(x)

        # Inject only to the highest resolution...
        u_inj = [self.stage0(x) if self.stage0 else x]
        for i in range(1, self.num_branches):
            bsz, _, H, W = u_inj[-1].shape
            u_inj.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2, device=x.device))   # ... and the rest are all zeros
        
        reset_norm(self.full_stage)
        z_now = [torch.zeros_like(each) for each in u_inj]
        
        def mdeq_func(*z_now):
            return self.full_stage(z_now, u_inj)

        # Multiscale Deep Equilibrium!
        z_out, info = self.deq(mdeq_func, z_now, solver_kwargs={'tau':self.args.tau})

        return z_out
 
    def forward(self, x, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)
