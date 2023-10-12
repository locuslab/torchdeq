# coding: utf-8
import argparse
import time
import math
import os, sys
import itertools
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from torchdeq.utils import add_deq_args

from data_utils import get_lm_corpus
from models.deq_transformer import DEQTransformerLM
from utils.exp_utils import create_exp_dir
from utils.data_parallel import BalancedDataParallel


parser = argparse.ArgumentParser(description='PyTorch DEQ Sequence Model')

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus (default to the WT103 path)')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--eval_n_layer', type=int, default=12,
                    help='number of total layers at evaluation')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads (default: 10)')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension (default: 50)')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension (default: match d_model)')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension (default: 500)')
parser.add_argument('--d_inner', type=int, default=8000,
                    help='inner dimension in the position-wise feedforward block (default: 8000)')

# Dropouts
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate (default: 0.05)')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention map dropout rate (default: 0.0)')

# Initializations
# Note: Generally, to make sure the DEQ model is stable initially, we should constrain the range
#       of initialization.
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.05,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')

# Optimizers
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='the number of steps to warm up the learning rate to its lr value')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')

# Gradient updates
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=200000,
                    help='upper epoch limit (at least 200K for WT103 or PTB)')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')

# Sequence logistics
parser.add_argument('--tgt_len', type=int, default=150,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=150,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--mem_len', type=int, default=150,
                    help='length of the retained previous heads')
parser.add_argument('--local_size', type=int, default=0,
                    help='local horizon size')

# DEQ techniques
add_deq_args(parser)

# Training techniques
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--eval', action='store_true',
                    help='evaluation mode')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--multi_gpu', action='store_true',
                    help='use multiple GPU')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default='LM', type=str,
                    help='experiment directory.')
parser.add_argument('--resume_dir', type=str, default='',
                    help='resume dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al. (Only 0 supported now)')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--resume_iter', type=int, default=0,
                    help='starting training step count (default to 0)')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--load_path', type=str, default='',
                    help='path to load weight')
parser.add_argument('--name', type=str, default='N/A',
                    help='name of the trial')

args = parser.parse_args()
args.tied = not args.not_tied

assert args.mem_len > 0, 'For now you must set mem_len > 0 when using deq'
    
if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.batch_size % args.batch_chunk == 0


if args.name == "N/A" and not args.debug:
    # If you find this too annoying, uncomment the following line and use time_stamp as name.
    # args.name = time_stamp
    raise ValueError("Please give a name to your run!")
print(f"Experiment name: {args.name}")


# Set up work dir
args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
time_stamp = time.strftime('%Y%m%d-%H%M%S')
model_path = time_stamp + f'-{args.name}'
args.work_dir = os.path.join(args.work_dir, model_path)

if args.resume_dir:
    args.work_dir = args.work_dir + '-resume-' + args.resume_dir.split('/')[-1]

logging = create_exp_dir(args.work_dir,
    scripts_to_save=['train.py', 'models/deq_transformer.py'], debug=args.debug)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load data
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = max(16, torch.cuda.device_count())
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len, device=device)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len, device=device)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len, device=device)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103']
    cutoffs = [20000, 40000, 200000]
    tie_projs += [True] * len(cutoffs)


# Build the model
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i].weight, 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('WeightShareSelfAttention') != -1:
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)


model = DEQTransformerLM(
        args, ntokens, args.n_layer, args.eval_n_layer, args.n_head, args.d_model, args.d_head, args.d_inner,
        args.dropout, args.dropatt, tie_weights=args.tied, d_embed=args.d_embed,
        div_val=args.div_val, tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, local_size=args.local_size,
        tgt_len=args.tgt_len, mem_len=args.mem_len, cutoffs=cutoffs, logging=logging
        )

if args.resume_dir:
    resume_path = os.path.join(args.resume_dir, 'model.pth')
    model.load_state_dict(torch.load(resume_path), strict=True)
    print(f'Load from {resume_path} ...')
elif args.load_path:
    model.load_state_dict(torch.load(args.load_path))
    print(f'Load from {args.load_path} ...')
else:
    model.apply(weights_init)    # Note: This applies weight_init recursively to modules in model
    model.word_emb.apply(weights_init)

args.n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

if args.multi_gpu:
    model = model.to(device)
    if args.gpu0_bsz >= 0 and args.batch_size != args.gpu0_bsz*torch.cuda.device_count():
        para_model = BalancedDataParallel(args.gpu0_bsz // args.batch_chunk, model).to(device)
    else:
        para_model = nn.DataParallel(model).to(device)
else:
    para_model = model.to(device)


#### optimizer
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if not args.debug and not args.eval:
    writer = SummaryWriter(log_dir=f'log/{args.dataset}/deq_{model_path}', flush_secs=5)
else:
    writer = None


#### scheduler
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)


# Monitor the backward pass
def write_backward_stats(info):
    for key in ('abs_lowest', 'rel_lowest', 'nstep'):
        writer.add_scalar(f'Backward/{key}', info[key].mean(), writer.global_steps)


# Resume from checkpoint
if args.resume_dir:
    # E.g., When you want to resume from a checkpoint from the same machine, where things should
    #       be stored in `args.resume_dir`
    if os.path.exists(os.path.join(args.resume_dir, 'optimizer.pth')):
        resume_opt_path = os.path.join(args.resume_dir, 'optimizer.pth')
        optimizer.load_state_dict(torch.load(resume_opt_path))
        print(f'Load from {resume_opt_path} ...')
    else:
        print('Optimizer was not saved. Start from scratch.')

if args.resume_iter > 0 and not args.resume_dir:
    # E.g., When you want to directly load a state_dict (e.g., trained on another machine), 
    #       You may want to manually adjust the optimizer's steps. On command line, you
    #       should run `bash ... --load [PATH] --resume_iter N` 
    #       in order to start the training from step N
    diff_from_warmup = args.resume_iter - args.warmup_step
    # Speed up the scheduler
    if args.scheduler in ['cosine', 'constant', 'dev_perf']:
        if diff_from_warmup < 0:
            # Hasn't finished warmup yet
            curr_lr = args.lr * args.resume_iter / args.warmup_step
            optimizer.param_groups[0]['lr'] = curr_lr
        else:
            if args.scheduler == 'cosine':
                for i in range(args.warmup_step, args.resume_iter):
                    optimizer.step()
                    scheduler.step(i)
    elif args.scheduler == 'inv_sqrt':
        for i in range(args.warmup_step, args.resume_iter):
            optimizer.step()
            scheduler.step(i)

logging('=' * 100)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 100)

###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    global train_step
    model.eval()

    # Evaluation
    total_len, total_loss = 0, 0
    abs_error, rel_error, counter = 0, 0, 0

    rho_list = []
    if args.sradius_mode:
        print("WARNING: You are evaluating with the power method at val. time. This may make things extremely slow.")
    
    with torch.no_grad():
        mems = []
        for i, (data, target, seq_len) in enumerate(eval_iter):
            if 0 < args.max_eval_steps <= i:
                break
            ret = para_model(data.t(), target.t(), mems, train_step=train_step,
                             b_max_iter=args.b_max_iter, compute_jac_loss=False,
                             sradius_mode=args.sradius_mode)
            loss, _, sradius, mems, info = ret
            loss = loss.mean()
            
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

            abs_error += info['abs_lowest'].mean().item()
            rel_error += info['rel_lowest'].mean().item()
            counter += 1

            if args.sradius_mode:
                rho_list.append(sradius.mean().item())
    
    if rho_list:
        logging(f"(Estimated) Spectral radius over validation set: {np.mean(rho_list)}")
    model.train()
    return total_loss / total_len, abs_error / counter, rel_error / counter


def train():
    global train_step, train_loss, train_jac_loss, best_val_loss, eval_start_time, log_start_time
    model.train()

    train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter

    if args.batch_chunk > 1:
        mems = [[] for _ in range(args.batch_chunk)]  # Each chunk (apparent) should have its own memory padding
    else:
        mems = []

    for batch, (data, target, seq_len) in enumerate(train_iter):
        if train_step < args.resume_iter:
            train_step += 1
            continue
        model.zero_grad()
        
        if train_step % 100 == 0:
            writer.global_steps = train_step
            writer_f = write_backward_stats
        else:
            writer_f = None

        # For DEQ:
        compute_jac_loss = np.random.uniform(0,1) < args.jac_loss_freq
        
        if args.batch_chunk > 1:
            # Mode 1: Using accumulated gradient to train on a larger (effective) batch size
            data_chunks = data.chunk(args.batch_chunk, dim=1)
            target_chunks = target.chunk(args.batch_chunk, dim=1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                ret = para_model(
                        data_i.t(), target_i.t(), mems[i], train_step=train_step, 
                        compute_jac_loss=compute_jac_loss, writer=writer_f
                        )
                loss, jac_loss, _, mems[i], info = ret         # mems[i]: # 3 x bsz
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                jac_loss = jac_loss.float().mean().type_as(loss) / args.batch_chunk
                
                if compute_jac_loss:
                    (loss + jac_loss * args.jac_loss_weight).backward()
                    train_jac_loss.append(jac_loss.float().item())
                else:
                    loss.backward()
                
                train_loss += loss.float().item()
        else:
            # Mode 2: Normal training with one batch per iteration
            ret = para_model(
                    data.t(), target.t(), mems, train_step=train_step, 
                    compute_jac_loss=compute_jac_loss, writer=writer_f
                    )
            loss, jac_loss, _, mems, info = ret
            loss = loss.float().mean().type_as(loss)
            jac_loss = jac_loss.float().mean().type_as(loss)

            if compute_jac_loss:
                (loss + jac_loss * args.jac_loss_weight).backward()
                train_jac_loss.append(jac_loss.float().item())
            else:
                loss.backward()

            train_loss += loss.float().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_step += 1

        # Step-wise learning rate annealing according to some scheduling (we ignore 'constant' scheduling)
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)

        # Logging of training progress
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            cur_ppl = math.exp(cur_loss)
            cur_jac_loss = np.mean(train_jac_loss)
            elapsed = time.time() - log_start_time
            cur_abs_error = info['abs_lowest'].mean().item()
            cur_rel_error = info['rel_lowest'].mean().item()
            cur_nstep = info['nstep'].mean().item()

            log_str = '| epoch {:3d} | step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                    '| ms/batch {:5.2f} | abs {:.3f} | rel {:.5f} | jac {:5.4f} | loss {:.2f} | ppl {:.3f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_abs_error, cur_rel_error,
                cur_jac_loss, cur_loss, cur_ppl)
            logging(log_str)

            train_loss = 0
            train_jac_loss = []
            log_start_time = time.time()

            if writer is not None:
                writer.add_scalar('Result/train_loss', cur_loss, train_step)
                writer.add_scalar('Result/train_ppl', cur_ppl, train_step)
                writer.add_scalar('Forward/abs', cur_abs_error, train_step)
                writer.add_scalar('Forward/rel', cur_rel_error, train_step)
                writer.add_scalar('Forward/nstep', cur_nstep, train_step)

        # Enter evaluation/inference mode once in a while and save the model if needed
        if train_step % args.eval_interval == 0:
            val_loss, eval_abs_error, eval_rel_error = evaluate(va_iter)
            val_ppl = math.exp(val_loss)
            logging('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                    '| abs {:.3f} | rel {:.5f} | valid loss {:5.2f} | valid ppl {:9.3f}'.format(
                train_step // args.eval_interval, train_step, (time.time() - eval_start_time), 
                eval_abs_error, eval_rel_error, val_loss, val_ppl)
            logging(log_str)
            logging('-' * 100)

            if writer is not None:
                writer.add_scalar('Result/valid_loss', val_loss, train_step)
                writer.add_scalar('Result/valid_ppl', val_ppl, train_step)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                ckpt_path = os.path.join(args.work_dir, 'model.pth')
                torch.save(model.state_dict(), ckpt_path)

                opt_path = os.path.join(args.work_dir, 'optimizer.pth')
                torch.save(optimizer.state_dict(), opt_path)
                
                best_val_loss = val_loss

            # dev-performance based learning rate annealing
            if args.scheduler == 'dev_perf':
                scheduler.step(val_loss)

            eval_start_time = time.time()
            
        if train_step == args.max_step:
            ckpt_path = os.path.join(args.work_dir, 'final.pth')
            torch.save(model.state_dict(), ckpt_path)
            break

        if args.jac_loss_weight > 0 and args.jac_loss_freq > 0 and \
           args.jac_incremental > 0 and train_step % args.jac_incremental == 0:
            logging(f"Adding 0.1 to jac. regularization weight after {train_step} steps")
            args.jac_loss_weight += 0.1

# Loop over epochs.
train_step = 0
train_loss = 0
train_jac_loss = []
best_val_loss = None

log_start_time = time.time()
eval_start_time = time.time()

if args.eval:
    train_step = 1e9
    epoch = -1
    
    eval_start_time = time.time()
    val_loss, eval_abs_error, eval_rel_error = evaluate(va_iter)
    val_ppl = math.exp(val_loss)
    logging('-' * 100)
    log_str = '| Eval | time: {:5.2f}s ' \
            '| abs {:.3f} | rel {:.5f} | valid loss {:5.2f} | valid ppl {:9.3f}'.format(
            (time.time() - eval_start_time), eval_abs_error, eval_rel_error, val_loss, val_ppl)
    logging(log_str)
    logging('-' * 100)
    
    eval_start_time = time.time()
    test_loss, eval_abs_error, eval_rel_error = evaluate(te_iter)
    test_ppl = math.exp(test_loss)
    logging('-' * 100)
    log_str = '| Test | time: {:5.2f}s ' \
            '| abs {:.3f} | rel {:.5f} | test loss {:5.2f} | test ppl {:9.3f}'.format(
            (time.time() - eval_start_time), eval_abs_error, eval_rel_error, test_loss, test_ppl)
    logging(log_str)
    logging('-' * 100)

    sys.exit(0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in itertools.count(start=1):
        train()
        if train_step == args.max_step:
            logging('-' * 100)
            logging('End of training')
            break
except KeyboardInterrupt:
    logging('-' * 100)
    logging('Exiting from training early')

# Load the best saved model.
best_path = os.path.join(args.work_dir, 'model.pth')
model.load_state_dict(torch.load(best_path))
para_model = model.to(device)

# Run on test data.
test_loss, eval_abs_error, eval_rel_error = evaluate(te_iter)
test_ppl = math.exp(test_loss)
logging('-' * 100)
log_str = '| Test | time: {:5.2f}s ' \
        '| abs {:.3f} | rel {:.5f} | test loss {:5.2f} | test ppl {:9.3f}'.format(
        (time.time() - eval_start_time), eval_abs_error, eval_rel_error, test_loss, test_ppl)
logging(log_str)
logging('-' * 100)


