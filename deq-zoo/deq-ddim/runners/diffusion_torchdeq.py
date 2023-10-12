import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

import PIL
from models.diffusion import Model

from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
import time 

import torchvision.utils as tvu
import wandb

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)

    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start

    elif beta_schedule == "geometric":
        ratio = 1 - beta_end
        betas = np.array([(ratio**n) for n in range(1, num_diffusion_timesteps+1)], dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0
            for i, (x, xT) in enumerate(train_loader):
                assert x.shape == xT.shape, f"Shape mismatch {x.shape} {xT.shape}"
                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.float().to(self.device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)

                b = self.betas
                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                if config.model.type == 'simple':
                    loss = loss_registry[config.model.type](model, x, t, e, b)
                elif 'modified' in config.model.type:
                    xT = xT.float().to(self.device)
                    loss = loss_registry[config.model.type](model, x, xT, t, e, b)

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch {epoch} step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))

                data_start = time.time()

    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)
            model.cuda()

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None
        else:
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            print("Loading checkpoint {}".format(ckpt))
            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model = torch.nn.DataParallel(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model, method=self.args.method)
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedure not defined")

    def reset_seed(self, seed):
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def sample_fid(self, model, method='anderson'):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        print(f"starting from image {img_id}")
        total_n_samples = 50000
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        total_time = 0
        with torch.no_grad():
            for round in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                if method == 'anderson':
                    all_xt = torch.repeat_interleave(x, self.args.timesteps, dim=0).to(x.device)
                    bsz, ch, h0, w0 = all_xt.shape
                    m = self.args.m
                    X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    H = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                    y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                    args = {
                        'all_xt': all_xt,
                        'bsz': x.size(0),
                    }
                    additional_args = self.get_additional_anderson_args(all_xt, xT=x, betas=self.betas, batch_size=x.size(0))
                    x = self.sample_image(x, model, args=args, additional_args=additional_args, method=method)

                elif method == 'ddpm':
                    all_xt = torch.repeat_interleave(x, self.args.timesteps, dim=0).to(x.device)
                    all_noiset = torch.randn_like(all_xt)

                    bsz, ch, h0, w0 = all_xt.shape
                    m = self.args.m
                    X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                    H = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                    y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                    args = {
                        'all_xt': all_xt,
                        'all_noiset': all_noiset,
                        'X': X,
                        'F': F,
                        'H': H,
                        'y': y,
                        'bsz': x.size(0),
                        'm': m,
                    }
                    additional_args = self.get_additional_anderson_args_ddpm(all_xt, xT=x, all_noiset=all_noiset, betas=self.betas, batch_size=x.size(0), eta=self.args.eta)
                    x = self.sample_image(x, model, args=args, additional_args=additional_args, method=method)
                else:
                    print("Method !!!!", method)
                    x = self.sample_image(x, model, method=method)

                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time += start.elapsed_time(end)

                if round % 50 == 0 or round == n_rounds - 1:
                    print(f"Round {round+1} Total Time {total_time} Avg time {total_time/(round+1)}")

                if type(x) == dict:
                    x_transformed = {}
                    for t in x.keys():
                        cur_img_idx = img_id
                        x_transformed[t] = inverse_data_transform(config, x[t])
                        for i in range(n):
                            tvu.save_image(
                                x_transformed[i], os.path.join(self.args.image_folder, str(t), "{cur_img_idx}.png")
                            )
                            cur_img_idx += 1
                else:
                    x = inverse_data_transform(config, x)

                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{img_id}.png")
                        )
                        img_id += 1

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def get_timestep_sequence(self):
        if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        elif self.args.skip_type == 'custom':
            seq = [s for s in range(0, 500, 2)] + [s for s in range(499, 1000, 4)] 
        else:
            raise NotImplementedError
        return seq

    def get_entire_timestep_sequence(self):
        seq = range(0, self.num_timesteps)
        return seq

    def sample_image(self, x, model, method, args=None, additional_args=None, last=True, sample_entire_seq=False):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            from functions.denoising import generalized_steps
            from functions.ddim_anderson import fp_implicit_iters_anderson
            from functions.ddpm_anderson import ddpm_implicit_iters_anderson

            logger=None
            if method == 'anderson':
                use_wandb = False
                if use_wandb:
                    wandb.init( project="DEQ-Efficiency-exp-rebuttal",
                                group=f"DDIM-xt-init-{self.config.data.dataset}-{self.config.data.category}-{self.args.method}-device{torch.cuda.device_count()}-{len(self.betas)}-{additional_args['T']}-m-{args['m']}-steps-15",
                                reinit=True,
                                config=self.config)
                    logger = wandb.log

                xs = fp_implicit_iters_anderson(x, model, self.betas, args=args, 
                               additional_args=additional_args, logger=logger, print_logs=True)
                if True or type(xs[0]) == dict:
                    xs = xs[0]
                    last = False

            elif method == 'simple-seq':
                use_wandb = False
                seq = self.get_timestep_sequence()
                if use_wandb:
                    wandb.init( project="DEQ-Efficiency-exps",
                                name=f"DDIM-{self.args.method}-device{torch.cuda.device_count()}-{len(self.betas)}-{len(seq)}",
                                reinit=True,
                                config=self.config)
                    logger = wandb.log
                xs = generalized_steps(x, seq, model, self.betas, logger=logger, print_logs=False, eta=self.args.eta)
            elif method == 'ddpm':
                use_wandb = False
                if use_wandb:
                    wandb.init( project="DDPM-exp-rebuttal",
                                group=f"DDPM-eta-{self.args.eta}-{self.config.data.dataset}-{self.config.data.category}-{self.args.method}-device{torch.cuda.device_count()}-{len(self.betas)}-{additional_args['T']}-m-{args['m']}-steps-15",
                                reinit=True,
                                config=self.config)
                    logger = wandb.log

                xs = ddpm_implicit_iters_anderson(x, model, self.betas, args=args, 
                              additional_args=additional_args, logger=logger, print_logs=True)
                if True or type(xs[0]) == dict:
                    xs = xs[0]
                    last = False
            else:
                if sample_entire_seq:
                    print("Sampling whole seq!")
                    seq = self.get_entire_timestep_sequence()
                else:
                    seq = self.get_timestep_sequence()
                xs = generalized_steps(x, seq, model, self.betas, logger=logger, print_logs=False, eta=self.args.eta)
            x = xs
            
        elif self.args.sample_type == "ddpm_noisy":
            seq = self.get_timestep_sequence()
            from functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def get_additional_anderson_args(self, all_xt, xT, betas, batch_size):
        from functions.ddim_anderson import compute_alpha
        seq = self.get_timestep_sequence()
        cur_seq = list(seq)
        seq_next = [-1] + list(seq[:-1])

        gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]
        xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]
        next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]
        prev_idx = [idx + 1 for idx in next_idx]

        plot_timesteps = []
        for batch_idx in range(batch_size):
            plot_timesteps += [n + batch_idx * len(cur_seq) for n in range(0, len(seq), len(seq)//10)] + [(batch_idx + 1) * len(cur_seq) - n for n in range(5, 0, -1)] 
        
        T = len(cur_seq)
        t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)
        next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        alpha_ratio = (at_next/at[0]).sqrt() 
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(all_xt.device)

        et_coeff2 = (1 - at_next).sqrt() - (((1 - at)*at_next)/at).sqrt()
        et_coeff = (1 / at_next.sqrt()) * et_coeff2
        et_prevsum_coeff = at_next.sqrt()
        
        additional_args = {
            "all_xT": all_xT, 
            "et_coeff": et_coeff,
            "et_prevsum_coeff": et_prevsum_coeff, 
            "T" : T, 
            "t" : t,
            "bz": batch_size,
            "plot_timesteps": plot_timesteps,
            "gather_idx": gather_idx,
            "xT_idx": xT_idx,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "xT": xT
        }
        return additional_args

    def get_additional_anderson_args_ddpm(self, all_xt, xT, all_noiset, betas, batch_size, **kwargs):
        from functions.ddim_anderson import compute_alpha
        seq = self.get_timestep_sequence()
        cur_seq = list(seq)
        seq_next = [-1] + list(seq[:-1])

        gather_idx = [idx for idx in range(len(cur_seq) - 1, len(all_xt), len(cur_seq))]
        xT_idx = [idx for idx in range(0, len(all_xt), len(cur_seq))]
        next_idx = [idx for idx in range(len(all_xt)) if idx not in range(len(cur_seq)-1, len(all_xt), len(cur_seq))]
        prev_idx = [idx + 1 for idx in next_idx]

        plot_timesteps = []
        for batch_idx in range(batch_size):
            plot_timesteps += [n + batch_idx * len(cur_seq) for n in range(0, len(seq), len(seq)//10)] + [(batch_idx + 1) * len(cur_seq) - n for n in range(5, 0, -1)] 
        
        T = len(cur_seq)
        t = torch.tensor(cur_seq[::-1]).repeat(batch_size).to(all_xt.device)
        next_t = torch.tensor(seq_next[::-1]).repeat(batch_size).to(all_xt.device)

        at = compute_alpha(betas, t.long())
        at_next = compute_alpha(betas, next_t.long())

        alpha_ratio = (at_next/at[0]).sqrt() 
        all_xT = alpha_ratio * torch.repeat_interleave(xT, T, dim=0).to(all_xt.device)

        if kwargs.get("eta", 1) == 0:
            raise ValueError("Running in DDPM mode but eta is 0")

        sigma_t = kwargs.get("eta", 1) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()

        ### This is c1(t) in the equations
        et_coeff2 = (1 - at_next - sigma_t**2).sqrt() - (((1 - at)*at_next)/at).sqrt()

        noise_t = (1 / at_next.sqrt()) * sigma_t * all_noiset


        et_coeff = (1 / at_next.sqrt()) * et_coeff2

        et_prevsum_coeff = at_next.sqrt()
        
        additional_args = {
            "all_xT": all_xT, 
            "et_coeff": et_coeff,
            "et_prevsum_coeff": et_prevsum_coeff, 
            "T" : T, 
            "t" : t,
            "bz": batch_size,
            "plot_timesteps": plot_timesteps,
            "gather_idx": gather_idx,
            "xT_idx": xT_idx,
            "prev_idx": prev_idx,
            "next_idx": next_idx,
            "xT": xT,
            "noise_t": noise_t
        }
        return additional_args
