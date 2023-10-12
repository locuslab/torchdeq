import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import configargparse
import yaml

import pickle
from tqdm import tqdm
import time

import numpy as np
import skvideo.io
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn

from torchdeq.utils import add_deq_args
from torchdeq.loss import fp_correction

from modules.modeling_utils import get_model, construct_model_args
from modules.utils import get_psnr_stats, batch_indices_generator, batched_apply


def preprocess(img_tensor):
    return img_tensor * 2 - 1


def postprocess(img_tensor):
    return torch.clamp(((img_tensor + 1) / 2), 0, 1).squeeze(-1).detach().cpu().numpy()


def hw2flat(hw_tensor):
    b, h, w, c = hw_tensor.shape
    flattened = hw_tensor.reshape(-1, c)
    return flattened, h, w


def flat2hw(flattened_tensor, h, w):
    c = flattened_tensor.size(1)
    hw = flattened_tensor.view(-1, h, w, c).contiguous()
    return hw


def train(args, train_data, model, opt=None, iters=10000, device='cuda', liveplot=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    train_in, train_tgt = train_data
    criterion = lambda x, y: torch.mean((x - y) ** 2)

    train_flattened_input, train_h, train_w = hw2flat(train_in)
    train_flattened_target, _, _ = hw2flat(preprocess(train_tgt))

    data_iter = tqdm(range(1, iters + 1))
    if run_label is not None:
        data_iter.set_description(run_label)

    log_writer = open(f'{args.log_dir}/logs.txt', 'w')
    
    step_list = []
    loss_list = []
    step_time_list = []
    test_psnr_list = []

    model = model.to(device)
    indices_generator = batch_indices_generator(len(train_flattened_input), args.train_batch_size, shuffle=True)

    pix_cache = torch.zeros(train_h * train_w, 1024)

    # Main training Loop
    postfix = {'loss': np.inf, 'psnr': 0., 'forward_steps': 0}
    for i in data_iter:
        start_time = time.time()
        
        switch_batch = (i - 1) % args.accelerated_training_iter == 0
        if switch_batch:
            iter_indices = sorted(next(indices_generator))
            z_init = torch.zeros(len(iter_indices), model.interm_channels).to(device)

        model_outputs = model(train_flattened_input[iter_indices].to(device), z_init, skip_solver=False, verbose=False)
        loss, fc_loss = fp_correction(
                criterion, (model_outputs['output'], train_flattened_target[iter_indices].to(device)),
                return_loss_values=True
                )

        loss_list.append(fc_loss[-1])

        if opt:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
            opt.step()
        
        postfix['forward_steps'] = model_outputs['forward_steps']
        postfix['loss'] = fc_loss[-1]
        
        step_time = time.time() - start_time

        do_log = i % args.log_freq == 0
        do_vis = i % args.vis_freq == 0

        if i % args.save_freq == 0:
            torch.save(model.state_dict(), f'{args.save_dir:s}/model_step{i:d}.pth')

        if do_log or do_vis:
            model.eval()
            with torch.no_grad():
                test_orig = train_tgt.numpy()
                test_rec = postprocess(flat2hw(
                    batched_apply(lambda x: model(x.to(device), torch.zeros(len(x), model.interm_channels).to(device)), 
                                 (train_flattened_input,),
                                 batch_size=args.test_batch_size,
                                 print_progress=True,
                        ), 
                    train_h, 
                    train_w
                ))
            model.train()

            if do_log:
                step_time_list.append(step_time)
                step_list.append(i)
                
                test_psnrs, test_psnr_mean, test_psnr_std = get_psnr_stats(test_orig, test_rec)
                test_psnr_list.append({
                    'mean': test_psnr_mean, 
                    'std': test_psnr_std, 
                    'psnr_per_frame': test_psnrs
                })

                log_writer.write(f'Iter {i}, loss: {loss.item():.5e}, PSNR mean: {test_psnr_mean:.3f}, PSNR std: {test_psnr_std:.3f}\n')
                log_writer.flush()
                
                postfix['psnr_mean'] = test_psnr_mean
                postfix['psnr_std'] = test_psnr_std

                summary_dict = {
                    'test_steps': step_list,
                    'test_psnr_stats': test_psnr_list,
                    'train_loss': loss_list,
                    'step_time': step_time_list
                }

                with open('{:s}/summary.pkl'.format(args.log_dir), 'wb') as summary_f:
                    pickle.dump(summary_dict, summary_f)

            if do_vis:
                inds = np.linspace(0, len(test_rec) - 1, 10).astype(np.int)

                fig, ax = plt.subplots(1, len(inds))
                fig.set_size_inches(4 * len(inds), 5)
                fig.set_tight_layout(True)
                for j, ind in enumerate(inds):
                    ax[j].imshow(test_rec[ind])
                    ax[j].set_axis_off()
                
                fig.savefig('{:s}/test_step{:d}.png'.format(args.vis_dir, i))
                test_vid = (test_rec * 255).astype(np.uint8)

                FPS = 25

                assert os.path.exists(args.vis_dir), 'Visualization directory {:s} does not exist'.format(args.vis_dir)
                writer = skvideo.io.FFmpegWriter(
                            '{:s}/test_step{:d}.mp4'.format(args.vis_dir, i),
                         )
                for j in range(len(test_vid)):
                    writer.writeFrame(test_vid[j])
                writer.close()

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

    model.eval()
    with torch.no_grad():
        flattened_input, h, w = hw2flat(test_input)
        test_rec = postprocess(flat2hw(
            batched_apply(lambda x: model(x.to(args.device), torch.zeros(len(x), model.z_channels).to(args.device)), 
                            (flattened_input,),
                            batch_size=args.test_batch_size,
                            print_progress=True,
                ), 
            h, 
            w
        ))
    model.train()

    _, psnr_mean, psnr_std = get_psnr_stats(test_rec, test_target.cpu().numpy())

    print("Test PSNR mean: {:.2f}, std: {:.2f}".format(psnr_mean, psnr_std))


def main(args):
    if args.dataset == 'cat':
        video_path = './data/video/cat_video.mp4'
    elif args.dataset == 'bikes':
        video_path = skvideo.datasets.bikes()
    
    data = torch.tensor(skvideo.io.vread(video_path).astype(np.single) / 255., dtype=torch.float32)

    t, h, w, c = data.shape

    t_arr = np.linspace(-1, 1, t, dtype=np.float32) 
    h_arr = np.linspace(-1, 1, h, dtype=np.float32)
    w_arr = np.linspace(-1, 1, w, dtype=np.float32)
    train_in = torch.tensor(np.stack(np.meshgrid(t_arr, h_arr, w_arr, indexing='ij'), axis=-1), dtype=torch.float32)

    model_args = construct_model_args(
        model_type=args.model_type,
        args=args,
        n_layers=args.n_layers, 
        in_channels=3,
        interm_channels=args.interm_channels, 
        output_channels=3,
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
    parser.add_argument('--dataset', default='cat', choices=['bikes', 'cat'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inference', default=False, action='store_true')

    parser.add_argument('--restore_path', default=None, type=str)

    parser.add_argument('--vis_freq', default=2000, type=int)
    parser.add_argument('--log_freq', default=2000, type=int)
    parser.add_argument('--save_freq', default=2000, type=int)

    parser.add_argument('--max_train_iters', default=10000, type=float)

    parser.add_argument('--model_type', default='implicit', choices=['implicit', 'siren', 'ffn'])
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--interm_channels', default=1024, type=int)
    parser.add_argument('--use_implicit', default=False, action='store_true')
    parser.add_argument('--input_scale', default=256., type=float)
    parser.add_argument('--filter_type', default='fourier', choices=['fourier', 'gabor', 'siren_like'])
    parser.add_argument('--gabor_alpha', default=3.)

    parser.add_argument('--accelerated_training_iter', default=1, type=int)

    parser.add_argument('--train_batch_size', default=256 * 256, type=int)
    parser.add_argument('--test_batch_size', default=256 * 256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    
    # Add args for utilizing DEQ
    add_deq_args(parser)
    args = parser.parse_args()

    args.log_dir = f'logs/videos/{args.prefix}/{args.experiment_id}/{args.dataset}'
    args.vis_dir = f'{args.log_dir}/visualizations'
    args.save_dir = f'{args.log_dir}/saved_models'

    [os.makedirs(path, exist_ok=True) for path in (args.log_dir, args.vis_dir, args.save_dir)]
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w+') as config_f:
        yaml.dump(vars(args), config_f)
 
    main(args)
