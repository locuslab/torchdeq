import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import livelossplot
import skimage
import configargparse
import yaml

from torchdeq.utils import add_deq_args

from modules.modeling_utils import get_model, construct_model_args, get_summary_dict
from modules.utils import get_psnr, network_spec


def load_dataset(filename, id):
    npz_data = np.load(filename)
    out = {
        "data_train":npz_data['train_data'] / 255.,
        "data_test":npz_data['test_data'] / 255.,
    }
    return out


def preprocess(img_tensor):
    return img_tensor * 2 - 1


def postprocess(img_tensor):
    return torch.clamp(((img_tensor + 1) / 2), 0, 1).squeeze(-1).detach().cpu().numpy()


def hw2batch(hw_tensor):
    b, h, w, c = hw_tensor.shape
    batched = hw_tensor.reshape(-1, c)
    return batched, h, w


def batch2hw(batched_tensor, h, w):
    c = batched_tensor.size(1)
    hw = batched_tensor.view(-1, h, w, c).contiguous()
    return hw


def train(args, train_data, test_data, model, opt=None, scheduler=None, iters=5000, device='cuda', visualize=False, vis_tag=None, vis_freq=100, log_freq=50, use_cached_input=False, run_label=None):
    """Standard training/evaluation epoch over the dataset"""
    train_in, train_tgt = train_data
    test_in, test_tgt = test_data
    criterion = lambda x, y: torch.mean((x - y) ** 2)

    train_batched_input, train_h, train_w = hw2batch(train_in)
    train_batched_target, _, _ = hw2batch(preprocess(train_tgt))

    test_batched_input, test_h, test_w = hw2batch(test_in)
    test_batched_target, _, _ = hw2batch(preprocess(test_tgt))

    data_iter = tqdm(range(iters), leave=True)
    if run_label is not None:
        data_iter.set_description(run_label)

    z_init = torch.zeros(len(train_batched_input), model.interm_channels).to(device)
    
    step_list = []
    loss_list = []
    step_time_list = []
    train_psnr_list = []
    test_psnr_list = []
    forward_step_list = []

    model = model.to(device)
    max_mem = 0

    # Main training Loop
    for i in data_iter:
        start_time = time.time()

        if args.batch_size < 0:
            batch_indices = np.arange(len(train_batched_input))
        else:
            batch_indices = np.random.choice(len(train_batched_input), args.batch_size, replace=False)

        model_outputs = model(train_batched_input[batch_indices], z_init[batch_indices], skip_solver=not args.no_skip_solver and i > 0, verbose=args.verbose)

        if use_cached_input:
            z_init[batch_indices] = model_outputs['imp_layer_output'].detach()
        
        forward_step_list.append(model_outputs['forward_steps'])

        loss = criterion(model_outputs['output'], train_batched_target[batch_indices])

        max_mem = max(max_mem, torch.cuda.memory_allocated(device))

        if opt:
            opt.zero_grad()
            loss.backward()

            opt.step()

            if scheduler:
                scheduler.step()

        step_time = time.time() - start_time
        step_time_list.append(step_time)

        do_visualize = (visualize and (i + 1) % vis_freq == 0)
        do_log = ((i + 1) % log_freq == 0)

        if do_log or do_visualize:
            model.eval()
            with torch.no_grad():
                train_orig = postprocess(batch2hw(train_batched_target, train_h, train_w))
                # train_rec = postprocess(batch2hw(model(train_batched_input, torch.zeros(1, model.interm_channels).to(device))['output'], train_h, train_w))
                train_rec = postprocess(batch2hw(model(train_batched_input)['output'], train_h, train_w))

                test_orig = postprocess(batch2hw(test_batched_target, test_h, test_w))
                # test_rec = postprocess(batch2hw(model(test_batched_input, torch.zeros(1, model.interm_channels).to(device))['output'], test_h, test_w))
                test_rec = postprocess(batch2hw(model(test_batched_input)['output'], test_h, test_w))
            model.train()

            if do_log:
                loss_list.append(loss.item())
                step_list.append(i + 1)

                train_psnr = get_psnr(train_orig, train_rec)
                train_psnr_list.append(train_psnr)
                    
                test_psnr = get_psnr(test_orig, test_rec)
                test_psnr_list.append(test_psnr)
                data_iter.set_postfix({'Train PSNR': train_psnr, 'Test PSNR': test_psnr})

            if do_visualize:
                fig, ax = plt.subplots(1, 4)
                fig.set_size_inches(16, 4)

                ax[0].clear()
                ax[0].set_axis_off()
                ax[0].imshow(train_orig[0], cmap='gray')
                ax[1].clear()
                ax[1].set_axis_off()
                ax[1].imshow(train_rec[0], cmap='gray')
                ax[2].clear()
                ax[2].set_axis_off()
                ax[2].imshow(test_orig[0], cmap='gray')
                ax[3].clear()
                ax[3].set_axis_off()
                ax[3].imshow(test_rec[0], cmap='gray')
                if vis_tag is None:
                    plt.show()
                    plt.close()
                else:
                    fig.savefig('{:s}/step_{:d}.jpeg'.format(args.vis_dir, i + 1), dpi=600)
    return {
        'loss': loss_list,
        'step_time': step_time_list,
        'step': step_list,
        'avg_forward_step': np.mean(forward_step_list[-1000:]),
        'train_psnr': train_psnr_list,
        'test_psnr': test_psnr_list,
        'max_mem': max_mem
        }


def render(model, render_input, device='cuda'):
    batched_input, h, w = hw2batch(render_input)
    rec = postprocess(batch2hw(model(batched_input)['output'], h, w))
    return rec


def main(args):
    if args.dataset == 'nature':
        # import div2k
        dataset = load_dataset('./data/image/data_div2k.npz', '1TtwlEDArhOMoH18aUyjIMSZ3WODFmUab')
        data_channels = 3
        RES = 512
    elif args.dataset == 'text':
        # import text
        dataset = load_dataset('./data/image/data_2d_text.npz', '1V-RQJcMuk9GD4JCUn70o7nwQE0hEzHoT')
        data_channels = 3
        RES = 512
    elif args.dataset == 'celeba':
        dataset = {
            'data_test': np.load('./data/image/celeba_128_tiny.npy').astype(np.float32).reshape(-1, 128, 128, 3)[:100] / 255
        }
        data_channels = 3
        RES = 128
    elif args.dataset == 'camera':
        # import cameraman
        camera_image = skimage.data.camera()
        dataset = {
            'data_test' : (camera_image.reshape(1, 512, 512, 1).astype(np.float32) / 255)
        }
        data_channels = 1
        RES = 512

    full_x = np.linspace(0, 1, RES) * 2 - 1
    full_x_grid = torch.tensor(np.stack(np.meshgrid(full_x,full_x), axis=-1)[None, :, :], dtype=torch.float32)

    if args.dataset in ['nature', 'text']:
        x_train = full_x_grid[:, ::2, ::2]
        x_test = full_x_grid[:, 1::2, 1::2]

        y_train = dataset['data_test'][:, ::2, ::2]
        y_test = dataset['data_test'][:, 1::2, 1::2]
    else:
        x_train = x_test = full_x_grid
        y_train = y_test = dataset['data_test']

    x_train, x_test = x_train.to(args.device), x_test.to(args.device)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(args.device), torch.tensor(y_test, dtype=torch.float32).to(args.device)

    summary_dict_path = '{:s}/{:s}_summary.pth'.format(args.log_dir, args.experiment_id)
    new_dict = get_summary_dict(args, 2, data_channels, args.filters, args.network_specs, args.input_scale)
    
    # Update config from checkpoints
    if args.continue_run and os.path.exists(summary_dict_path):
        print("Restoring summary dict from {:s}".format(os.path.abspath(summary_dict_path)))
        summary_dict = torch.load(summary_dict_path)
        for k in new_dict:
            if k not in summary_dict:
                summary_dict[k] = new_dict[k]
    else:
        summary_dict = new_dict

    log_writer = open('{:s}/logs.txt'.format(args.log_dir), 'w+' if not args.continue_run else 'a+')

    for model_id in summary_dict:
        model_dict = summary_dict[model_id]
        if model_dict['finished']:
            print("Skipping finished model {:s}".format(model_id))
            continue

        model_dict['misc']['imgs'] = []
        model_dict['misc']['test_time'] = []
        model_dict['results']['train_psnr'] = []
        model_dict['results']['test_psnr'] = []
        model_dict['results']['train_step_time'] = []
        model_dict['results']['avg_forward_step'] = []
        max_train_mem = 0
        for i in range(len(y_train)):
            torch.cuda.empty_cache()

            def compute_lr_decay(t):
                return (1 - (1 - args.lr_decay) * t / args.max_train_iters)

            model = get_model(model_dict['config']).to(args.device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, compute_lr_decay)
            if i == 0: print(model)

            train_stats = train(
                args,
                (x_train, y_train[i].unsqueeze(0)),
                (x_test, y_test[i].unsqueeze(0)),
                model,
                opt,
                scheduler,
                log_freq=args.log_freq,
                iters=args.max_train_iters,
                device=args.device,
                use_cached_input=not model_dict['config']['one_pass'],
                run_label=f'{model_id}/Image {i}'
            )

            max_train_mem = max(max_train_mem, train_stats['max_mem'])
            model_dict['results']['train_psnr'].append(train_stats['train_psnr'])
            model_dict['results']['test_psnr'].append(train_stats['test_psnr'])
            model_dict['results']['train_step_time'].append(train_stats['step_time'])
            model_dict['results']['avg_forward_step'].append(train_stats['avg_forward_step'])

            model_dict['state_dicts'].append(model.state_dict())
            
            model.eval()
            with torch.no_grad():
                full_input = full_x_grid.to(args.device)
                start_time = time.time()
                img = render(model, full_input, args.device)
                elapsed_time = time.time() - start_time
            model.train()

            model_dict['misc']['imgs'].append(img)
            model_dict['misc']['test_time'].append(elapsed_time)

        model_dict['misc']['param_count'] = sum([p.numel() for p in model.parameters()])
        model_dict['misc']['train_max_memory_allocated'] = max_train_mem
        print(model_dict['misc']['train_max_memory_allocated'] / (1024 ** 3))

        train_psnr_final = np.array(model_dict['results']['train_psnr'], ndmin=2)[:, -1]
        test_psnr_final = np.array(model_dict['results']['test_psnr'], ndmin=2)[:, -1]

        train_psnr_mean, train_psnr_std = np.mean(train_psnr_final), np.std(train_psnr_final)
        test_psnr_mean, test_psnr_std = np.mean(test_psnr_final), np.std(test_psnr_final)
        avg_forward_step_mean, avg_forward_step_std = np.mean(model_dict['results']['avg_forward_step']), np.std(model_dict['results']['avg_forward_step'])

        log_writer.write("Model: {:s} Dataset: {:s} Train mean: {:.4f}, std: {:.4f}; Test mean: {:.4f}, std: {:.4f}; Avg forward steps mean: {:.2f}, std: {:.2f}\n"\
            .format(model_id, args.dataset, train_psnr_mean, train_psnr_std, test_psnr_mean, test_psnr_std, avg_forward_step_mean, avg_forward_step_std))
        log_writer.flush()
        
        model_dict['misc']['train_stat'] = (train_psnr_mean, train_psnr_std)
        model_dict['misc']['test_stat'] = (test_psnr_mean, test_psnr_std)
        model_dict['finished'] = True
        torch.save(summary_dict, summary_dict_path)
    
    log_writer.close()
    

if __name__ == '__main__':
    parser = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--prefix', default='vanilla')
    parser.add_argument('--experiment_id', default='vanilla')
    parser.add_argument('-c', '--config_file', default=None, is_config_file=True)
    parser.add_argument('--dataset', default='nature', choices=['camera', 'celeba', 'nature', 'text'])
    parser.add_argument('--log_dir', default=None, type=str)

    parser.add_argument("--batch_size", default=-1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=1., type=float)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--filters', default=['fourier', 'gabor', 'siren_like'], nargs='+', type=str)
    parser.add_argument('--network_specs', default=[[1, 256], [1, 512], [4, 256]], nargs='+', type=network_spec)
    parser.add_argument('--input_scale', default=256., type=float)
    parser.add_argument('--no_skip_solver', default=False, action='store_true')

    parser.add_argument('--max_train_iters', default=2000, type=int)
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--verbose', default=False, action='store_true')

    parser.add_argument('--continue_run', default=False, action='store_true')
    
    # Add args for utilizing DEQ
    add_deq_args(parser)
    args = parser.parse_args()

    if args.log_dir is None:
        args.log_dir = f'logs/images/{args.prefix}/{args.experiment_id}/{args.dataset}'
    
    [os.makedirs(path, exist_ok=True) for path in (args.log_dir,)]
    with open(os.path.join(args.log_dir, 'config.yaml'), 'w+') as config_f:
        yaml.dump(vars(args), config_f)

    main(args)
