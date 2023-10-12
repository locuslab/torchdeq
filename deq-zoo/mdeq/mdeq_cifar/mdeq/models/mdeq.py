from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools
from termcolor import colored

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

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

                
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        A bottleneck block with receptive field only 3x3. (This is not used in MDEQ; only
        in the classifier layer).
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM, affine=False)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion, momentum=BN_MOMENTUM, affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x) + injection
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

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
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.fuse_method = fuse_method
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

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels):
        """
        To check if the config file is consistent
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
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


class MDEQ(nn.Module):
    def __init__(self, cfg, args, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters
        """
        super().__init__()

        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)

        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
                                        nn.ReLU())
        
        # PART II: MDEQ's f_\theta layer
        self.full_stage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']      
        num_channels = self.num_channels
        block = blocks_dict[self.full_stage_cfg['BLOCK']]
        self.full_stage = self._make_stage(self.full_stage_cfg, num_channels)
        
        # PART III: Call TorchDEQ.
        apply_norm(self.full_stage, args=args)
        self.deq = get_deq(args)

        # PART IV: Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(self.num_channels)
        self.classifier = nn.Linear(self.final_chansize, self.num_classes)
            
    def parse_cfg(self, cfg):
        global DEQ_EXPAND, NUM_GROUPS
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
 
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]

        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']

        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        
        self.head_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['HEAD_CHANNELS']
        self.final_chansize = cfg['MODEL']['EXTRA']['FULL_STAGE']['FINAL_CHANSIZE']
           
    def _make_stage(self, layer_config, num_channels):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, fuse_method)
    
    def _make_head(self, pre_stage_channels):
        """
        Create a classification head that:
           - Increase the number of features in each resolution 
           - Downsample higher-resolution equilibria to the lowest-resolution and concatenate
           - Pass through a final FC layer for classification
        """
        head_block = Bottleneck
        d_model = self.init_chansize
        head_channels = self.head_channels
        
        # Increasing the number of channels on each resolution when doing classification. 
        incre_modules = []
        for i, channels  in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], blocks=1, stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)
            
        # Downsample the high-resolution streams to perform classification
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(conv3x3(in_channels, out_channels, stride=2, bias=True),
                                            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                                            nn.ReLU(inplace=True))
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        # Final FC layers
        final_layer = nn.Sequential(nn.Conv2d(head_channels[len(pre_stage_channels)-1] * head_block.expansion,
                                              self.final_chansize,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0),
                                    nn.BatchNorm2d(self.final_chansize, momentum=BN_MOMENTUM),
                                    nn.ReLU(inplace=True))
        return incre_modules, downsamp_modules, final_layer

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def decode(self, y_list):
        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](y_list[i+1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)

        # Pool to a 1x1 vector (if needed)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        y = self.classifier(y)
        return y

    def forward(self, x, train_step=-1, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
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
        z_out, info = self.deq(mdeq_func, z_now, writer=kwargs.get('writer', None))
        if train_step > 0 and train_step % 100 == 0:
            print(info['rel_lowest'].mean().item(), info['abs_lowest'].mean().item(), info['nstep'].mean().item())

        if self.training:
            y_out = [self.decode(each) for each in z_out]
            return y_out
        else:
            y = self.decode(z_out[-1])
            return y
 
    def init_weights(self, pretrained='',):
        """
        Model initialization. If pretrained weights are specified, we load the weights.
        """
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d) and m.weight is not None:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
 
            
def get_cls_net(config, args, **kwargs):
    global BN_MOMENTUM
    BN_MOMENTUM = 0.1
    model = MDEQ(config, args, **kwargs)
    model.init_weights()
    return model


