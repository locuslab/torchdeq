from logging import log
import torch
import wandb
import time 


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def compute_multi_step(xt, all_xT, model, et_coeff, et_prevsum_coeff, T, t, image_dim, xT, **kwargs):
    with torch.no_grad():
        xt_all = torch.zeros_like(all_xT)
        xt_all[kwargs['xT_idx']] = xT
        xt_all[kwargs['prev_idx']] = xt[kwargs['next_idx']]

        xt = xt_all.to('cuda')

        et = model(xt, t)
        et_updated = et_coeff * et
        et_cumsum_all = et_updated.cumsum(dim=0)
        et_prevsum = et_cumsum_all

        all_seqs = torch.arange(T-1, et_cumsum_all.shape[0]-1, T)
        prev_cumsum = 0
        if len(all_seqs) > 0:
            for idx in all_seqs:
                prev_cumsum += torch.unsqueeze(et_cumsum_all[idx], dim=0)
                et_prevsum[idx+1:idx+1+T] -= torch.repeat_interleave(prev_cumsum, T,  dim=0)

        xt_next = all_xT + et_prevsum_coeff * et_prevsum
        log_dict = {
            "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
            "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
            "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
        }
    return xt_next.view(xt.shape[0], -1), log_dict


@torch.no_grad()
def anderson(f, x0, X, F, H, y, args, m=5, lam=1e-3, max_iter=15, tol=1e-2, beta = 1.0, logger=None):
    """ Anderson acceleration for fixed point iteration. """
    with torch.no_grad():
        bsz, ch, h0, w0 = x0.shape
        
        t1 = time.time()

        X[:,0] = x0.view(bsz, -1)
        F[:,0], _ = f(xt=x0.view(x0.shape), **args)

        X[:,1] = F[:,0].view(bsz, -1)
        F[:,1], _ = f(xt=F[:,0].view(x0.shape), **args)

        H[:,0,1:] = H[:,1:,0] = 1
        y[:,0] = 1
        
        t2 = time.time()
        print("Intial set up", t2-t1)
        time_logger = {
            "setup": t2 - t1,
            "bmm": 0,
            "solve": 0,
            "forward call-unet": 0,
            "total_time_per_iter": 0
        }

        iter_count = 0
        log_metrics = {}
        res = []
        norm_res = []
        for k in range(2, max_iter):
            n_ = min(k, m)
            G = F[:,:n_]-X[:,:n_]
            
            t3 = time.time()
            H[:,1:n_+1,1:n_+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n_, dtype=x0.dtype,device=x0.device)[None]
            t4 = time.time()
            alpha = torch.solve(y[:,:n_+1], H[:,:n_+1,:n_+1])[0][:, 1:n_+1, 0]   # (bsz x n)
            t5 = time.time()
            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n_])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n_])[:,0]
            F[:,k%m], log_metrics = f(xt=X[:,k%m].view(x0.shape), **args)
            t6 = time.time()

            residual = (F[:,k%m] - X[:,k%m]).norm().item()
            normalized_residual = (F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm()).item()

            t7 = time.time()
            res.append(residual)
            norm_res.append(normalized_residual)
            
            time_logger["bmm"] += (t4 - t3)
            time_logger["solve"] += (t5 - t4)
            time_logger["forward call-unet"] += (t6 - t5)
            time_logger["total_time_per_iter"] += (t7 - t3)
            iter_count += 1

            ### TODO: break out early for norm_res
            if (norm_res[-1] < tol):
                print("Breaking out early at {}".format(k))
                break

            #print("{}/{} Residual {} tol {} ".format(k, max_iter, res[-1], tol))
            if logger is not None:
                log_metrics["residual"] = residual
                log_metrics["normalized_residual"] = normalized_residual

                log_metrics["alpha"] = torch.norm(alpha, dim=-1).mean()
                log_metrics["samples"] = [wandb.Image(X[:, k%m].view_as(x0).to('cpu')[ts]) for ts in args['plot_timesteps']]
                log_metrics["setup"] = time_logger['setup']
                log_metrics["total_time_per_iter"] = time_logger['total_time_per_iter'] / iter_count
                log_metrics["total_time"] = t7 - t1
                for key, val in time_logger.items():
                    if key not in log_metrics: 
                        log_metrics[f"avg-{key}"] = val / iter_count
                        log_metrics[key] = val
                log_metrics["perc_time_forward_call"] = time_logger["forward call-unet"] * 100 / time_logger["total_time_per_iter"]
                logger(log_metrics)
    x_eq = X[:,k%m].view_as(x0)[args['gather_idx']].to('cpu')
    return x_eq


def fp_implicit_iters_anderson(x, model, b, args=None, additional_args=None, logger=None, print_logs=False, save_last=True, **kwargs):
    with torch.no_grad():
        x0_preds = []
        image_dim = x.shape
        
        all_xt = args['all_xt']
        additional_args["model"] = model
        additional_args["image_dim"] = image_dim

        x_final = anderson(compute_multi_step, all_xt, args['X'], args['F'], args['H'], args['y'], 
                                 additional_args, m=args['m'], lam=1e-3, max_iter=15, tol=1e-3, beta = 1.0, logger=logger)
    return x_final, x0_preds

