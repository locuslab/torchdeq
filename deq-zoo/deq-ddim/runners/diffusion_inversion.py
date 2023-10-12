import os

from runners.diffusion import Diffusion

import numpy as np
import tqdm
import torch

from models.diffusion import Model

from models.ema import EMAHelper
from functions import get_optimizer
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from functions.denoising import forward_steps, generalized_steps
import time 

import torchvision.utils as tvu
import wandb


from functions.latent_space_opt_sddim import DEQLatentSpaceOpt


class DiffusionInversion(Diffusion):
    
    def __init__(self, args, config, device=None):
        super().__init__(args, config, device)

    # latent space optimization
    def ls_opt(self):

        # Do initial setup
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger
        
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)

        # First, I need to get my data!!!
        dataset, _ = get_dataset(args, config)
        
        # Load model in eval mode!
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
        for param in model.parameters():
            param.requires_grad = False

        B = 1
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        seq = self.get_timestep_sequence()

        global_time = 0
        global_min_l2_dist = 0
        # epsilon value for early stopping
        eps = 0.5
        img_idx = 0
        for _ in range(self.config.ls_opt.num_samples):

            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            start_epoch = 0

            if args.use_wandb:
                run = wandb.init(project="latent-space-opt-final", reinit=True, name=f"trial-{args.seed}",
                            group=f"{config.data.dataset}-{config.data.category}-DDPM-indistr-{self.config.ls_opt.in_distr}-T{args.timesteps}-parallel-{self.config.ls_opt.use_parallel}-" +
                                f"l1-{self.args.lambda1}-l2-{self.args.lambda2}-l3-{self.args.lambda3}-lr-{config.optim.lr}-" + 
                                 f"-devices-2",
                                #f"-devices-{torch.cuda.device_count()}",
                            settings=wandb.Settings(start_method="fork"),
                            config=args
                            )
            if self.config.ls_opt.in_distr:
                with torch.no_grad():
                    x_target = torch.randn(
                        B,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device
                    )
                    x_target = self.sample_image(x_target.detach().view((B, C, H, W)), model, method="generalized")

            else:
                img_idx = np.random.randint(low=0, high=len(dataset))
                x_init, _ = dataset[img_idx]
                x_target = x_init.view(1, C, H, W).float().cuda()
                x_target = data_transform(self.config, x_target)

            if self.config.ls_opt.use_parallel:
                x = torch.randn(
                    B,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device 
                )
                # Smart initialization for faster convergence
                # This further improves results from paper by a lot!
                with torch.no_grad():
                    all_x, _ = forward_steps(x_target, seq, model, self.betas)
                    x = all_x[-1].detach().clone()

                # This ensures that this gradient descent updates can be performed on this  
                all_xt = torch.repeat_interleave(x, self.args.timesteps+1, dim=0).to(x.device).requires_grad_() 
                
                if self.config.ls_opt.method == 'ddpm':
                    print("Performing optimization on DDPM!!!")
                    sddim = True
                else:
                    sddim = False
                    
                deq_ddim = DEQLatentSpaceOpt(args, model, sddim=sddim)
                
                if self.config.ls_opt.method == 'ddpm':
                    eta = self.args.eta
                    all_noiset = torch.randn(
                        self.args.timesteps * B,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device 
                    ).to(x.device)
                else:
                    eta = 0.
                    all_noiset = None

                diffusion_args = deq_ddim.get_ddim_injection(
                        all_xt, seq, self.betas, x.size(0),
                        eta, all_noiset
                        )

                optimizer = get_optimizer(self.config, [all_xt])
                min_loss = float('inf')
                best_img_src = x
                min_l2_dist = float('inf')

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for epoch in range(start_epoch, config.training.n_epochs):
                    optimizer.zero_grad()

                    xt_pred = deq_ddim(
                            all_xt, diffusion_args, 
                            logger=None)
                    
                    loss_target = (xt_pred[-1] - x_target).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    loss_reg = all_xt[0].detach().square().sum()

                    loss = args.lambda1 * loss_target

                    loss.backward()
                    optimizer.step()

                    if loss < min_loss:
                        print("Min loss encountered!")
                        min_loss = loss
                        best_img_src = all_xt[0].detach().clone()
                        min_l2_dist = loss_target
                    
                    log_image = loss < eps
                    if args.use_wandb and (epoch % config.training.snapshot_freq == 0 or epoch == 0 or epoch == 1 or epoch == config.training.n_epochs-1) or log_image:
                        with torch.no_grad():
                            
                            best_img_src = best_img_src.view(B, C, H, W)
                            cur_img_latent = torch.repeat_interleave(best_img_src, self.args.timesteps, dim=0).to(x.device)
                            if self.config.ls_opt.method == 'ddpm':

                                bsz, ch, h0, w0 = cur_img_latent.shape
                                # m = self.args.m
                                m = 5
                                X = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                F = torch.zeros(bsz, m, ch * h0 * w0, dtype=all_xt.dtype, device=all_xt.device)
                                H_and = torch.zeros(bsz, m+1, m+1, dtype=all_xt.dtype, device=all_xt.device)
                                y = torch.zeros(bsz, m+1, 1, dtype=all_xt.dtype, device=all_xt.device)

                                sampling_args = {
                                    'all_xt': cur_img_latent,
                                    'all_noiset': all_noiset,
                                    'X': X,
                                    'F': F,
                                    'H': H_and,
                                    'y': y,
                                    'bsz': x.size(0),
                                    'm': m,
                                }
                                sampling_additional_args = self.get_additional_anderson_args_ddpm(cur_img_latent, 
                                                        xT=best_img_src, 
                                                        all_noiset=all_noiset, 
                                                        betas=self.betas, 
                                                        batch_size=x.size(0), 
                                                        eta=self.args.eta)

                                generated_image = self.sample_image(x=cur_img_latent, model=model, 
                                                        args=sampling_args,
                                                        additional_args=sampling_additional_args,
                                                        method="ddpm")
                            else:
                                generated_image = self.sample_image(best_img_src.view((B, C, H, W)), model, method="generalized")
                            
                            generated_image = inverse_data_transform(config, generated_image)

                            logged_images = [
                                wandb.Image(x_target.detach().squeeze().view((C, H, W))),
                                wandb.Image(generated_image.detach().squeeze().view((C, H, W))),
                            ]
                            wandb.log({
                                    "all_images": logged_images
                                    })
                    print(f"Epoch {epoch}/{config.training.n_epochs} Loss {loss} xT {torch.norm(all_xt[0][-1])} dist {loss_target} " +
                                    f"reg {loss_reg}")
                    
                    if args.use_wandb:
                        log_dict = {
                            "Loss": loss.item(),
                            "max all_xt": all_xt.max(),
                            "min all_xt": all_xt.min(),
                            "mean all_xt": all_xt.mean(),
                            "std all_xt": all_xt.std(),
                            "all_xt grad norm": all_xt.grad.norm(),
                            "dist ||x_0 - x*||^2": loss_target.item(),
                            "reg ||x_T||^2": loss_reg.item(),
                        }

                        wandb.log(log_dict)

                    if loss < eps:
                        print(f"Early stopping! Breaking out of loop at {epoch}")
                        break

                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time = start.elapsed_time(end)

                if args.use_wandb:
                    log_dict = {
                        "min L2 dist": min_l2_dist.item(),
                        "min_loss": min_loss.item(),
                        "total time": total_time
                    }
                    wandb.log(log_dict)

                for i in range(B):
                    generated_image = self.sample_image(best_img_src.view((B, C, H, W)), model, method="generalized")
                    generated_image = inverse_data_transform(config, generated_image)
                    tvu.save_image(
                        generated_image[i], os.path.join(args.image_folder, f"anderson-gen-{img_idx}.png")
                    )
                    x_target = inverse_data_transform(config, x_target)
                    tvu.save_image(
                        x_target, os.path.join(args.image_folder, f"anderson-target-{img_idx}.png")
                    )
            else:
                # You can start with random initialization
                # This is much difficult case but also slower
                # x = torch.randn(
                #     B,
                #     config.data.channels,
                #     config.data.image_size,
                #     config.data.image_size,
                #     device=self.device 
                # ).requires_grad_()

                # Smart initialization for faster convergence
                # This further improves results from paper by a lot!
                with torch.no_grad():
                    all_x, _ = forward_steps(x_target, seq, model, self.betas)
                    x = all_x[-1].detach().clone()
                
                x = x.requires_grad_()

                optimizer = get_optimizer(self.config, [x])
                
                min_loss = float('inf')
                best_img_src = x
                min_l2_dist = float('inf')

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                for epoch in range(start_epoch, config.training.n_epochs):
                    optimizer.zero_grad()

                    xs, _ = generalized_steps(x, seq, model, self.betas, logger=None, print_logs=False, eta=self.args.eta)

                    loss_target = (xs[-1] - x_target).square().sum(dim=(1, 2, 3)).mean(dim=0)
                    loss_reg = x.detach().square().sum()
                    loss = self.args.lambda1 * loss_target

                    loss.backward()
                    optimizer.step()
                    
                    if loss < min_loss:
                        min_loss = loss
                        best_img_src = xs[-1]
                        min_l2_dist = loss_target
                    
                    log_image = loss < eps
                    if args.use_wandb and ((epoch == 0 or epoch == config.training.n_epochs-1) or log_image):
                        with torch.no_grad():
                            generated_image = self.sample_image(x.detach().view((B, C, H, W)), model, method="generalized", sample_entire_seq=False)

                            logged_images = [
                                wandb.Image(x_target.detach().squeeze().view((C, H, W))),
                                wandb.Image(generated_image.detach().squeeze().view((C, H, W)))
                            ] #+ [wandb.Image(xs[i].detach().view((C, H, W))) for i in range(0, len(xs), len(xs)//10)]
                            wandb.log({
                                    "all_images": logged_images
                                    })

                    print(f"Epoch {epoch}/{self.config.training.n_epochs} Loss {loss} xT {torch.norm(x)} dist {loss_target} reg {loss_reg}")
                    
                    if args.use_wandb:
                        log_dict = {
                            "Loss": loss.item(),
                            "max all_xt": x.max(),
                            "min all_xt": x.min(),
                            "mean all_xt": x.mean(),
                            "std all_xt": x.std(),
                            "x grad norm": x.grad.norm(),
                            "dist ||x_0 - x*||^2": loss_target.item(),
                            "reg ||x_T||^2": loss_reg.item(),
                        }
                        wandb.log(log_dict)
                    
                    if loss < eps:
                        print(f"Early stopping! Breaking out of loop at {epoch}")
                        break

                end.record()
                # Waits for everything to finish running
                torch.cuda.synchronize()
                total_time = start.elapsed_time(end)

                if args.use_wandb:
                    log_dict = {
                        "min L2 dist": min_l2_dist.item(),
                        "min_loss": min_loss.item(),
                        "total time": total_time
                    }
                    wandb.log(log_dict)

                for i in range(B):
                    generated_image = self.sample_image(x.detach().view((B, C, H, W)), model, method="generalized")
                    generated_image = inverse_data_transform(config, generated_image)
                    tvu.save_image(
                        generated_image[i], os.path.join(self.args.image_folder, f"seq-gen-{img_idx}.png")
                    )
                    x_target = inverse_data_transform(config, x_target)
                    tvu.save_image(
                        x_target, os.path.join(self.args.image_folder, f"seq-target-{img_idx}.png")
                    )

            print("Summary stats for anderson acceleration")
            print(f"Average time {total_time/(epoch+1)}")
            print(f"Min l2 dist {min_l2_dist}")

            if args.use_wandb:
                run.finish()

            global_time += total_time
            global_min_l2_dist += min_l2_dist
            
            print(f"Current Overall Time    : {global_time/self.config.ls_opt.num_samples}")
            print(f"Current Overall L2 dist : {min_l2_dist/self.config.ls_opt.num_samples}")
        
            torch.cuda.empty_cache()

        print(f"Overall Time    : {global_time/self.config.ls_opt.num_samples}")
        print(f"Overall L2 dist : {min_l2_dist/self.config.ls_opt.num_samples}")
        
        try:
            stats = torch.load('./stats')
        except:
            stats = {'time':[],
                    'loss':[]}
        
        stats['time'].append(global_time/self.config.ls_opt.num_samples)
        stats['loss'].append(min_l2_dist.item()/self.config.ls_opt.num_samples)
        torch.save(stats, './stats')

    def reconstruction(self):
        args, config = self.args, self.config

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

        # First, I need to get my data!!!
        dataset, _ = get_dataset(args, config)
        B = 1
        C, H, W = config.data.channels, config.data.image_size, config.data.image_size
        
        seq = self.get_timestep_sequence()
        from functions.denoising import forward_steps
        model.eval()
        for _ in range(self.config.ls_opt.num_samples):
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            if args.use_wandb:
                run = wandb.init(project="latent-space-opt-final", reinit=True, name=f"trial-{args.seed}",
                            group=f"{config.data.dataset}-{config.data.category}-DDIM-recons-T{args.timesteps}-parallel-{self.config.ls_opt.use_parallel}-" +
                                f"-devices-{torch.cuda.device_count()}",
                            settings=wandb.Settings(start_method="fork"),
                            config=args
                            )
            img_idx = np.random.randint(low=0, high=len(dataset))
            x_init, _ = dataset[img_idx]
            x_target = x_init.view(1, C, H, W).float().cuda()
            x_target = data_transform(self.config, x_target)

            with torch.no_grad():
                all_x, _ = forward_steps(x_target, seq, model, self.betas)
                latent_src = all_x[-1]
                generated_image = self.sample_image(latent_src.view((B, C, H, W)), model, method="generalized")
                loss = (generated_image - x_target).square().sum(dim=(1, 2, 3)).mean(dim=0)
                generated_image = inverse_data_transform(config, generated_image)

                if args.use_wandb:
                    logged_images = [
                        wandb.Image(x_target.detach().squeeze().view((C, H, W))),
                        wandb.Image(latent_src.detach().squeeze().view((C, H, W))),
                        wandb.Image(generated_image.detach().squeeze().view((C, H, W))),
                    ]
                    wandb.log({
                            "all_images": logged_images,
                            "loss": loss
                            })
