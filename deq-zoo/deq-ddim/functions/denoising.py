import torch
from functions.utils import get_ortho_mat
import wandb
import time 

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def compute_cumprod_alpha(beta):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0)
    return a


def get_alpha_at_index(a, t):
    a_val = a.index_select(0, t + 1).view(-1, 1, 1, 1)
    return a_val


def generalized_steps(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    bsz = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    alpha = compute_cumprod_alpha(b)

    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(bsz) * i).to(x.device)
        next_t = (torch.ones(bsz) * j).to(x.device)
        
        at = get_alpha_at_index(alpha, t.long())
        at_next = get_alpha_at_index(alpha, next_t.long())
        
        xt = xs[-1].to('cuda')

        et = model(xt, t)

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

        x0_preds.append(x0_t)
        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        noise_t = torch.randn_like(x)
        xt_next = at_next.sqrt() * x0_t + c1 * noise_t + c2 * et
        xs.append(xt_next)
    return xs, x0_preds


def forward_steps(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    bsz = x.size(0)
    prev_seq = list(seq[:-1])
    seq_next = list(seq[1:])
    x0_preds = []
    xs = [x]
    alpha = compute_cumprod_alpha(b)

    for i, j in zip(prev_seq, seq_next):
        t = (torch.ones(bsz) * i).to(x.device)
        next_t = (torch.ones(bsz) * j).to(x.device)
        
        at = get_alpha_at_index(alpha, t.long())
        at_next = get_alpha_at_index(alpha, next_t.long())
        
        xt = xs[-1].to('cuda')
        et = model(xt, t)
        
        coeff_et = (1/at_next - 1).sqrt() - (1/at - 1).sqrt()
        coeff_xt = (1/at).sqrt() - (1/at_next).sqrt()

        xt_next =xt + at_next.sqrt() * (coeff_xt * xt + coeff_et * et)
        xs.append(xt_next)
    return xs, x0_preds


def generalized_steps_fp_ddim(x, seq, model, b, logger=None, print_logs=False, **kwargs):
    with torch.no_grad():
        B = x.size(0)
        x0_preds = []
        xs = [x]

        image_dim = x.shape
        T = seq[-1]
        diff = seq[-1] - seq[-2]

        for i in range(T, -1, -diff):
            t = (torch.ones(B) * i).to(x.device)

            next_t = max(-1, i-diff)
            next_t = (torch.ones(B) * next_t).to(x.device)

            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            
            et_coeff = ((1 - at_next)).sqrt()

            xt_next = at_next.sqrt() * x0_t + et_coeff * et

            log_dict = {
                    "alpha at": torch.mean(at),
                    "alpha at_next": torch.mean(at_next),
                    "xt": torch.norm(xt.reshape(image_dim[0], -1), -1).mean(),
                    "xt_next": torch.norm(xt_next.reshape(image_dim[0], -1), -1).mean(),
                    "coeff x0": at_next.sqrt().squeeze().mean(),
                    "coeff et": et_coeff.squeeze().mean(),
                    "prediction et": torch.norm(et.reshape(image_dim[0], -1), -1).mean(),
                }
            
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(xt_next[i]) for i in range(10)]
                logger(log_dict)
            elif print_logs:
                print(t, max(-1, i-diff), log_dict)

            xs.append(xt_next.view(image_dim).to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, logger=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        image_dim = x.shape
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))

            log_dict = {}
            if logger is not None:
                if i % 50 == 0 or i < 50:
                    log_dict["samples"] = [wandb.Image(x[i]) for i in range(0, 1000, 100)]
                logger(log_dict)
    return xs, x0_preds
