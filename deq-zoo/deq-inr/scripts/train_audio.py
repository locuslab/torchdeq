import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import configargparse
import yaml

import pickle
from tqdm import tqdm
import time

import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn

from torchdeq.utils import add_deq_args

from modules.modeling_utils import get_model, construct_model_args
from modules.utils import get_psnr, batch_indices_generator, batched_apply


def preprocess(in_tensor):
    return in_tensor.unsqueeze(-1)


def postprocess(in_tensor):
    return in_tensor.squeeze(-1).detach().cpu().numpy()


def get_psnr_stats(output, target):
    psnr_list = []
    for i in range(len(output)):
        psnr_list.append(get_psnr(output[i], target[i]))
    
    psnr_mean = np.mean(psnr_list)
    psnr_std = np.std(psnr_list)

    return psnr_list, psnr_mean, psnr_std


def train(args, train_data, model, opt, iters=10000, device='cuda', liveplot=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    train_in, train_tgt = train_data
    train_in = preprocess(train_in).to(device)
    train_tgt = train_tgt.to(device)

    criterion = lambda x, y: torch.mean((x - y) ** 2)

    data_iter = tqdm(range(1, iters + 1))
    if run_label is not None:
        data_iter.set_description(run_label)

    log_writer = open(f'{args.log_dir}/logs.txt', 'w')
    
    step_list = []
    loss_list = []
    step_time_list = []
    test_psnr_list = []

    model = model.to(device)

    splits = np.linspace(0, len(train_in), args.grad_accum_steps + 1).astype(np.int)
    z_inits = [torch.zeros(splits[i + 1] - splits[i], model.interm_channels).to(device) for i in range(len(splits) - 1)]
    # Main training Loop
    postfix = {'loss': np.inf, 'psnr': 0., 'forward_steps': 0}
    torch.autograd.set_detect_anomaly(True)
    for i in data_iter:
        start_time = time.time()
        
        opt.zero_grad()
        
        for j in range(len(splits) - 1):
            model_outputs = model(train_in[splits[j]: splits[j + 1]], z_inits[j], skip_solver=i > 1, verbose=False)
            if args.use_implicit and 'imp_layer_output' in model_outputs:
                z_inits[j] = model_outputs['imp_layer_output'].detach()

            loss = criterion(model_outputs['output'].squeeze(-1), train_tgt[splits[j]: splits[j + 1]])
            loss_list.append(loss.item())

            loss.backward()

        opt.step()
        
        if 'forward_steps' in model_outputs:
            postfix['forward_steps'] = model_outputs['forward_steps']
        postfix['loss'] = loss.item()
        
        step_time = time.time() - start_time

        do_log = i % args.log_freq == 0
        do_vis = i % args.vis_freq == 0

        if i % args.save_freq == 0:
            torch.save(model.state_dict(), f'{args.save_dir:s}/model_step{i:d}.pth')

        if do_log or do_vis:
            model.eval()
            with torch.no_grad():
                test_orig = train_tgt.cpu().numpy() * args.audio_amp
                test_rec = postprocess(model(train_in)['output']) * args.audio_amp
            model.train()

            if do_log:
                step_time_list.append(step_time)
                step_list.append(i)
                
                test_psnr = get_psnr(test_orig, test_rec)
                test_psnr_list.append(test_psnr)

                log_writer.write(f'Iter {i}, loss: {loss.item():.5e}, PSNR {test_psnr:.2f}\n')
                log_writer.flush()

                postfix['psnr'] = test_psnr
                
                summary_dict = {
                    'test_steps': step_list,
                    'test_psnr_stats': test_psnr_list,
                    'train_loss': loss_list,
                    'step_time': step_time_list
                }

                with open('{:s}/summary.pkl'.format(args.log_dir), 'wb') as summary_f:
                    pickle.dump(summary_dict, summary_f)

            if do_vis:
                fig, axes = plt.subplots(1, 3)
                fig.set_size_inches(24, 3)
                fig.tight_layout(pad=0.2)
                axes[0].plot(np.linspace(0, 1, len(test_orig)), test_orig)
                axes[1].plot(np.linspace(0, 1, len(test_rec)), test_rec)
                axes[2].plot(np.linspace(0, 1, len(test_orig)), test_orig - test_rec)

                for _ax in axes:
                    _ax.set_ylim(-1.05, 1.05)

                axes[1].get_yaxis().set_ticks([])
                axes[2].get_yaxis().set_ticks([])

                axes[0]

                fig.savefig('{:s}/test_step{:d}.png'.format(args.vis_dir, i))

                scipy.io.wavfile.write('{:s}/test_step{:d}.wav'.format(args.vis_dir, i), args.audio_rate, test_rec)
                np.save('{:s}/test_step{:d}_source.npy'.format(args.vis_dir, i), test_rec)
        data_iter.set_postfix(postfix)
    
    log_writer.close()

    return {
        'loss': loss_list,
        'step_time': step_time_list,
        'step': step_list,
        'test_psnr': test_psnr_list,
        }


def test(args, test_data, model):
    assert args.restore_path is not None, 'Restore path cannot be empty'

    test_input, test_target = test_data
    model.to(args.device)

    with torch.no_grad():
        test_rec = postprocess(model(test_input.unsqueeze(-1).to(args.device), verbose=True)['output'])

    psnr = get_psnr(test_rec, test_target.cpu().numpy())

    print("Test PSNR {:2f}".format(psnr))

def main(args):
    datasets = {
        'counting': './data/audio/gt_counting.wav',
        'bach': './data/audio/gt_bach.wav'
    }

    audio_path = datasets[args.dataset]
    rate, data = scipy.io.wavfile.read(audio_path)
    if len(data.shape) > 1 and data.shape[1] == 2:
        data = np.mean(data, axis=1)

    # Record maximum audio amplitude
    args.audio_amp = np.abs(np.max(data))
    args.audio_rate = rate
    data = torch.tensor(data / args.audio_amp, dtype=torch.float32)

    print('Sampling rate: {:d}, Amplitude: {:f}'.format(rate, args.audio_amp))

    t_arr = np.linspace(-1., 1., len(data), dtype=np.float32)
    train_in = torch.tensor(t_arr, dtype=torch.float32)

    model_args = construct_model_args(
        model_type=args.model_type,
        args=args,
        n_layers=args.n_layers, 
        in_channels=1,
        interm_channels=args.interm_channels, 
        output_channels=1,
        input_scale=args.input_scale,
        use_implicit=args.use_implicit, 
        filter_type=args.filter_type,
        filter_options={'alpha': args.gabor_alpha},
        norm_type=args.norm_type,
    )
    model = get_model(model_args)
    print(model)

    if args.restore_path is not None:
        model.load_state_dict(torch.load(args.restore_path))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.inference:
        train(
            args,
            (train_in, data),
            model,
            opt,
            iters=args.max_train_iters,
            device=args.device
        )
    else:
        test(
            args,
            (train_in, data),
            model
        )


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--prefix', default='vanilla', type=str)
    parser.add_argument('--experiment_id', default='vanilla', type=str)
    parser.add_argument('-c', '--config_file', default=None, is_config_file=True)
    parser.add_argument('--dataset', default='bach', choices=['counting', 'bach'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference', default=False, action='store_true')

    parser.add_argument('--restore_path', default=None, type=str)

    parser.add_argument('--max_train_iters', default=3000, type=int)

    parser.add_argument('--vis_freq', default=500, type=int)
    parser.add_argument('--log_freq', default=500, type=int)
    parser.add_argument('--save_freq', default=500, type=int)

    parser.add_argument('--grad_accum_steps', default=1, type=int, 
                        help="Number of steps used for gradient accumulation. "
                             "Set to >1 when data cannot into the device memory")

    parser.add_argument('--model_type', default='implicit', choices=['implicit', 'siren', 'ffn'])
    parser.add_argument('--n_layers', default=4, type=int)
    parser.add_argument('--interm_channels', default=128, type=int)
    parser.add_argument('--use_implicit', default=False, action='store_true')
    parser.add_argument('--input_scale', default=25000., type=float)
    parser.add_argument('--filter_type', default='fourier', choices=['fourier', 'gabor', 'siren_like'])
    parser.add_argument('--gabor_alpha', default=3., type=float)

    parser.add_argument('--lr', default=1e-3, type=float)
    
    # Add args for utilizing DEQ
    add_deq_args(parser)
    args = parser.parse_args()

    args.log_dir = f'logs/audios/{args.prefix}/{args.experiment_id}/{args.dataset}'
    args.vis_dir = f'{args.log_dir}/visualizations'
    args.save_dir = f'{args.log_dir}/saved_models'
    [os.makedirs(path, exist_ok=True) for path in (args.log_dir, args.vis_dir, args.save_dir)]
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w+') as config_f:
        yaml.dump(vars(args), config_f)

    main(args)
