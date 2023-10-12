# Modified based on the HRNet repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torchdeq.utils import add_deq_args
from torchprofile import profile_macs

import _init_paths
import models
from config import config
from config import update_config
from core.cls_function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from termcolor import colored


def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')

    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('--eval', action='store_true', help="Enable Eval mode.")

    parser.add_argument('--repeat', '-r',
                        help='number of repeat times for experiments',
                        default=1,
                        type=int)

    parser.add_argument('--exp_id', help='model directory', type=str, default='cifar_mdeq')
    parser.add_argument('--log_dir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--data_dir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--test_model',
                        help='testModel',
                        type=str,
                        default='')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    # Add args for utilizing DEQ
    add_deq_args(parser)

    args = parser.parse_args()
    update_config(config, args)

    return args


def train_model(args, logger, final_output_dir, tb_log_dir):
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config, args).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).cuda()
    model(dump_input)
    macs = profile_macs(model, dump_input)
    params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    logger.info(f'Params (M): {params/1e6:.3f}, MACs (G): {macs/1e9:.4f}')

    if config.TRAIN.MODEL_FILE:
        model.load_state_dict(torch.load(config.TRAIN.MODEL_FILE))
        logger.info(colored('=> loading model from {}'.format(config.TRAIN.MODEL_FILE), 'red'))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    print("Finished constructing model!")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = get_optimizer(config, model)
    lr_scheduler = None

    print(optimizer)

    best_perf = 0.0
    best_model = False
    last_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_perf = checkpoint['perf']
            model.module.load_state_dict(checkpoint['state_dict'])

            # Update weight decay if needed
            checkpoint['optimizer']['param_groups'][0]['weight_decay'] = config.TRAIN.WD
            optimizer.load_state_dict(checkpoint['optimizer'])

            if 'lr_scheduler' in checkpoint:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1e5,
                                  last_epoch=checkpoint['lr_scheduler']['last_epoch'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            best_model = True

    # Data loading code
    dataset_name = config.DATASET.DATASET

    if dataset_name == 'imagenet':
        traindir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TRAIN_SET)
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 is supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        augment_list = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if config.DATASET.AUGMENT else []
        transform_train = transforms.Compose(augment_list + [
            transforms.ToTensor(),
            normalize,
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=True, download=True, transform=transform_train)
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # Learning rate scheduler
    if lr_scheduler is None:
        if config.TRAIN.LR_SCHEDULER != 'step':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(train_loader)*config.TRAIN.END_EPOCH, eta_min=1e-6)
        elif isinstance(config.TRAIN.LR_STEP, list):
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR,
                last_epoch-1)

    # Training code
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        topk = (1,5) if dataset_name == 'imagenet' else (1,)
        if config.TRAIN.LR_SCHEDULER == 'step':
            lr_scheduler.step()

        # train for one epoch
        train(args, config, train_loader, model, criterion, optimizer, lr_scheduler, epoch,
              final_output_dir, tb_log_dir, writer_dict, topk=topk)
        torch.cuda.empty_cache()

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, model, criterion, lr_scheduler, epoch,
                                  final_output_dir, tb_log_dir, writer_dict, topk=topk)
        torch.cuda.empty_cache()
        writer_dict['writer'].flush()

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': config.MODEL.NAME,
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }, best_model, final_output_dir, filename='checkpoint.pth.tar')

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()

    logger.info('=> best acc {}'.format(best_perf))

    return best_perf.item()


def eval_model(args, logger, final_output_dir, tb_log_dir):
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(config, args).cuda()

    dump_input = torch.rand(config.TRAIN.BATCH_SIZE_PER_GPU, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]).cuda()
    params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
   
    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, 'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    
    gpus = list(config.GPUS)
    model = nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # Data loading code
    dataset_name = config.DATASET.DATASET

    topk = (1,5) if dataset_name == 'imagenet' else (1,)
    if dataset_name == 'imagenet':
        valdir = os.path.join(config.DATASET.ROOT+'/images', config.DATASET.TEST_SET)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_valid = transforms.Compose([
            transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.ImageFolder(valdir, transform_valid)
    else:
        assert dataset_name == "cifar10", "Only CIFAR-10 and ImageNet are supported at this phase"
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # For reference
        
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        valid_dataset = datasets.CIFAR10(root=f'{config.DATASET.ROOT}', train=False, download=True, transform=transform_valid)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    perf = validate(config, valid_loader, model, criterion, None, epoch=-1, output_dir=final_output_dir,
             tb_log_dir=tb_log_dir, writer_dict=None, topk=topk)

    return perf.item()


if __name__ == '__main__':
    args = parse_args()
    logger, final_output_dir, tb_log_dir, log_path = create_logger(
        args, config, args.cfg, 'train')
    
    # copy model file
    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(final_output_dir, 'models')
    if os.path.exists(models_dst_dir):
        shutil.rmtree(models_dst_dir)
    shutil.copytree(os.path.join(this_dir, 'mdeq/'), models_dst_dir)

    if args.eval:
        perf = eval_model(args, logger, final_output_dir, tb_log_dir)
        print(perf)

        eval_f_max_iter = args.eval_f_max_iter
        for f_max_iter in range(eval_f_max_iter):
            args.eval_f_max_iter = f_max_iter
            perf = eval_model(args, logger, final_output_dir, tb_log_dir)
            print(f_max_iter, perf)
    else:
        stats_path = log_path + '.stats.csv'

        steps_list = [args.grad[-1]]
        stats_dict = dict()
        for steps in steps_list:
            best_rec = list()
            for _ in range(args.repeat):
                best_acc = train_model(args, logger, final_output_dir, tb_log_dir)
                best_rec.append(best_acc)

            stats_dict[steps] = best_rec

            df = pd.DataFrame.from_dict(stats_dict, orient='index')
            df = df.sort_index()
            df.to_csv(stats_path)
            print('Current steps:', steps)
            print(df)
