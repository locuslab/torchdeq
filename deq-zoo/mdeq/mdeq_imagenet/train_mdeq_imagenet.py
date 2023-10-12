import os, argparse, time, warnings, torch, torchvision
import numpy as np
from os.path import join, isfile

# DALI data reader
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

# distributed
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

# TorchDEQ
from torchdeq.utils import add_deq_args
from torchdeq.loss import fp_correction

# Models
from models.mdeq import MDEQClsNet
from utils import Logger, save_checkpoint, AverageMeter, accuracy, CosAnnealingLR
from tensorboardX import SummaryWriter


warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.simplefilter('error')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--eval', action='store_true', help='Evaluation mode.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--data', metavar='DIR', default="./data", help='path to dataset')

parser.add_argument('--num-classes', default=1000, type=int, metavar='N', help='Number of classes')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.1, help="learning rate")

parser.add_argument('--nesterov', action='store_true', help='Use Nesterov SGD optimizer')
parser.add_argument('--epochs', type=int, default=120, help="epochs")
parser.add_argument('--wd', '--weight-decay', type=float, default=1e-4, help="weight decay")

parser.add_argument('--imsize', default=256, type=int, metavar='N', help='image resize size')
parser.add_argument('--imcrop', default=224, type=int, metavar='N', help='image crop size')
parser.add_argument('--warmup', default=5, type=int, help="warmup epochs")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')

parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to the latest checkpoint')
parser.add_argument('--save-path', default="results/tmp", type=str, help='path to save results')

# Add args for utilizing DEQ
add_deq_args(parser)

# distributed
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--dali-cpu', action='store_true', help='Runs CPU based version of DALI pipeline.')
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

os.makedirs(args.save_path, exist_ok=True)
if args.local_rank == 0:
    tfboard_writer = SummaryWriter(log_dir=args.save_path)
    logger = Logger(join(args.save_path, "log.txt"))

    this_dir = os.path.dirname(__file__)
    models_dst_dir = os.path.join(args.save_path, 'models')
    os.system(f'cp -r models {args.save_path}')


# DALI pipelines
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        # MXnet rec reader
        self.input = ops.MXNetReader(
                path=join(data_dir, "train.rec"),
                index_path=join(data_dir, "train.idx"),
                random_shuffle=True,
                shard_id=args.local_rank,
                num_shards=args.world_size
        )
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(
                device=decoder_device, output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100
        )
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255]
        )
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(
                path=join(data_dir, "val.rec"),
                index_path=join(data_dir, "val.idx"),
                random_shuffle=False,
                shard_id=args.local_rank,
                num_shards=args.world_size
        )
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(
                device="gpu",
                output_dtype=types.FLOAT,
                output_layout=types.NCHW,
                crop=(crop, crop),
                image_type=types.RGB,
                mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                std=[0.229 * 255,0.224 * 255,0.225 * 255]
        )
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]  


def main():
    if args.local_rank == 0:
        logger.info(args)

    # Pytorch distributed setup
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()

    # Build DALI dataloader
    pipe = HybridTrainPipe(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=args.local_rank,
            data_dir=args.data, 
            crop=args.imcrop,
            dali_cpu=args.dali_cpu
    )
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(
            batch_size=50,
            num_threads=args.workers,
            device_id=args.local_rank,
            data_dir=args.data,
            crop=args.imcrop,
            size=args.imsize
    )
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    train_loader_len = int(train_loader._size / args.batch_size)

    # Define model and optimizer
    device = torch.device('cuda', args.local_rank)
    model = MDEQClsNet(args).to(device)

    if args.local_rank == 0:
        logger.info("Model details:")
        logger.info(model)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.SGD(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
            nesterov=args.nesterov
    )
    if args.local_rank == 0:
        logger.info("Optimizer details:")
        logger.info(optimizer)

    # Optionally resume from a checkpoint
    if args.resume is not None:
        assert isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        if args.local_rank == 0:
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        if args.local_rank == 0: 
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        
        train_step = args.start_epoch * train_loader_len
        acc1, acc5 = validate(val_loader, model, train_step)
        torch.cuda.empty_cache()

        if args.eval:
            return acc1

        if 'best_acc1' in checkpoint:
            best_acc1 = checkpoint['best_acc1']
        else:
            best_acc1 = acc1

        if args.local_rank == 0:
            # Write to tfboard
            tfboard_writer.add_scalar('test/acc1', acc1, args.start_epoch-1)
            tfboard_writer.add_scalar('test/acc5', acc5, args.start_epoch-1)
            tfboard_writer.flush()

    else:
        args.start_epoch = 0
        if args.local_rank == 0: best_acc1 = 0    

    # Define learning rate scheduler
    scheduler = CosAnnealingLR(
            loader_len=train_loader_len,
            epochs=args.epochs,
            lr_max=args.lr,
            lr_min=1e-6,
            warmup_epochs=args.warmup,
            last_epoch=args.start_epoch-1
    )

    for epoch in range(args.start_epoch, args.epochs):
        # Train and evaluate
        train(train_loader, model, optimizer, scheduler, epoch)
        torch.cuda.empty_cache()

        acc1, acc5 = validate(val_loader, model, scheduler.iter_counter)
        torch.cuda.empty_cache()

        if args.local_rank == 0:
            # Write to tfboard
            tfboard_writer.add_scalar('test/acc1', acc1, epoch)
            tfboard_writer.add_scalar('test/acc5', acc5, epoch)
            tfboard_writer.flush()

            # Remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
            logger.info("Best acc1=%.5f" % best_acc1)

            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict()
                }, is_best, path=args.save_path, filename="checkpoint.pth")
    
    return best_acc1


def train(train_loader, model, optimizer, scheduler, epoch):
    if args.local_rank == 0:
        data_times, batch_times, losses, acc1, acc5 = [AverageMeter() for _ in range(5)]
        train_loader_len = int(np.ceil(train_loader._size/args.batch_size))

    # Switch to train mode
    model.train()
    if args.local_rank == 0:
        end = time.time()
    for i, data in enumerate(train_loader):
        # Load data and distribute to devices
        image = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        if args.local_rank == 0:
            start = time.time()

        # Compute the learning rate
        lr = scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward training image and compute cross entropy loss
        pred = model(image, train_step=scheduler.iter_counter)
        loss = fp_correction(F.cross_entropy, (pred, target), gamma=args.gamma)

        # One SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        top1, top5 = accuracy(pred[-1], target, topk=(1, 5))

        # Gather tensors from different devices
        loss = reduce_tensor(loss)
        top1 = reduce_tensor(top1)
        top5 = reduce_tensor(top5)

        # Update AverageMeter stats
        if args.local_rank == 0:
            data_times.update(start - end)
            batch_times.update(time.time() - start)
            losses.update(loss.item(), image.size(0))
            acc1.update(top1.item(), image.size(0))
            acc5.update(top5.item(), image.size(0))
        # torch.cuda.synchronize()

        if args.local_rank == 0 and scheduler.iter_counter == 630000:
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'iters': scheduler.iter_counter, 
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, True, path=args.save_path, filename="pretrain-checkpoint.pth")

        # Log training info
        if i % args.print_freq == 0 and args.local_rank == 0:
            tfboard_writer.add_scalar("train/learning-rate", lr, epoch*train_loader_len+i)
            logger.info(
                    'Ep[{0}/{1}] It[{2}/{3}] Bt {batch_time.avg:.3f} Dt {data_time.avg:.3f} '
                    'Loss {loss.val:.3f} ({loss.avg:.3f}) Acc1 {top1.val:.3f} ({top1.avg:.3f}) '
                    'Acc5 {top5.val:.3f} ({top5.avg:.3f}) LR {4:.3E}'.format(
                            epoch, args.epochs, i, train_loader_len, lr,
                            batch_time=batch_times, data_time=data_times,
                            loss=losses, top1=acc1, top5=acc5
                    )
            )
        if args.local_rank == 0:
            end = time.time()

    # Reset the training loader
    train_loader.reset()
    if args.local_rank == 0:
        tfboard_writer.add_scalar('train/loss', losses.avg, epoch)
        tfboard_writer.add_scalar('train/acc1', acc1.avg, epoch)
        tfboard_writer.add_scalar('train/acc5', acc5.avg, epoch)


@torch.no_grad()
def validate(val_loader, model, train_step):
    if args.local_rank == 0:
        losses, top1, top5 = [AverageMeter() for _ in range(3)]
        val_loader_len = int(np.ceil(val_loader._size/50))

    # Switch to evaluate mode
    model.eval()
    for i, data in enumerate(val_loader):
        image = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()

        # Compute output
        prediction = model(image, train_step=train_step)
        loss = F.cross_entropy(prediction, target, reduction='mean')

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(prediction, target, topk=(1, 5))

        # Gather tensors from different devices
        loss = reduce_tensor(loss)
        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)

        # Update meters and log info
        if args.local_rank == 0:
            losses.update(loss.item(), image.size(0))
            top1.update(acc1.item(), image.size(0))
            top5.update(acc5.item(), image.size(0))
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}] Test Loss {loss.val:.3f} (avg={loss.avg:.3f}) '
                            'Acc1 {top1.val:.3f} (avg={top1.avg:.3f}) Acc5 {top5.val:.3f} (avg={top5.avg:.3f})' \
                            .format(i, val_loader_len, loss=losses, top1=top1, top5=top5))
    if args.local_rank == 0:
        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'.format(top1=top1, top5=top5))

    # Reset the validation loader
    val_loader.reset()
    top1 = torch.tensor([top1.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    top5 = torch.tensor([top5.avg]).cuda() if args.local_rank == 0 else torch.tensor([0.]).cuda()
    dist.broadcast(top1, 0)
    dist.broadcast(top5, 0)
    return top1.cpu().item(), top5.cpu().item()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.reduce(rt, 0, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == '__main__':
    main()
    if args.local_rank == 0:
        tfboard_writer.close()
        logger.close()
