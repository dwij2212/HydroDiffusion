import torch
import pdb
import math

def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    """
    Cosine schedule for log-SNR (Nichol & Dhariwal, 2021).
    t: tensor in [0,1], any shape.
    Returns log-SNR of same shape.
    """
    # ensure tensors live on the same device
    device = t.device
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max, device=device)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min, device=device))) - b
    return -2.0 * torch.log(torch.tan(a * t + b))

def diffusion_params(t):
    logsnr = logsnr_schedule_cosine(t)
    alpha  = torch.sqrt(torch.sigmoid(logsnr))[..., None]
    sigma  = torch.sqrt(torch.sigmoid(-logsnr))[..., None]
    return logsnr, alpha, sigma


def get_beta_schedule(num_steps=1000, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, num_steps)
    
    
def q_sample(x_start, t, noise, betas):
    '''
    Forward diffusion:
      x_start: [B, H]
      noise:   [B, H]
      t:       [B] long
      betas:   [T]
    '''
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    a_bar_t = alpha_cumprod[t].unsqueeze(1).to(x_start.device)        # [B,1]
    one_minus_a_bar = (1.0 - alpha_cumprod[t]).unsqueeze(1).to(x_start.device)
    return a_bar_t.sqrt() * x_start + one_minus_a_bar.sqrt() * noise

def compute_posterior_mean(x_start, noise, t, betas):
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)

    a_t        = alphas[t]                  # [B]
    a_bar_t    = alpha_cumprod[t]             # [B]
    idx_prev   = (t - 1).clamp(min=0)
    a_bar_prev = alpha_cumprod[idx_prev]     # [B]

    a_bar_t_sqrt = a_bar_t.sqrt().unsqueeze(1)            
    one_minus_a_bar = (1 - a_bar_t).sqrt().unsqueeze(1)
    x_t = a_bar_t_sqrt * x_start + one_minus_a_bar * noise

    coef1 = (a_bar_prev.sqrt() * betas[t] / (1 - a_bar_t)).unsqueeze(1)
    coef2 = (a_t.sqrt() * (1 - a_bar_prev) / (1 - a_bar_t)).unsqueeze(1)

    mu_tilde = coef1 * x_start + coef2 * x_t
    return mu_tilde

def get_diffusion_schedules(beta_start=1e-4, beta_end=0.02, num_steps=1000, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "T": num_steps,
        "sqrt_alpha_cumprod": alpha_cumprod.sqrt(),
        "sqrt_one_minus_alpha_cumprod": (1 - alpha_cumprod).sqrt(),
        "sqrt_recip_alpha": (1.0 / alphas).sqrt(),
        "posterior_variance": betas[1:] * (1.0 - alpha_cumprod[:-1]) / (1.0 - alpha_cumprod[1:])
    }

def p_sample(model, x_t, t, schedules, static_attrs, x_cond, guidance_weight, future_precip):
    betas = schedules["betas"]
    sqrt_recip_alpha = schedules["sqrt_recip_alpha"]
    sqrt_one_minus_alpha_cumprod = schedules["sqrt_one_minus_alpha_cumprod"]

    # Call guided_forward instead of unconditional forward
    eps_theta = model.guided_forward(x_cond, t, static_attrs, future_precip, guidance_weight)  # [B, H]

    mean = (
        sqrt_recip_alpha[t].view(-1, 1) *
        (x_t - betas[t].view(-1, 1) * eps_theta / sqrt_one_minus_alpha_cumprod[t].view(-1, 1))
    )

    if t[0] == 0:
        return mean
    else:
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(betas[t].view(-1, 1)) * noise

def sample(model, shape, schedules, device, static_attrs, x_cond, 
           num_steps=1000, guidance_weight=2.0, future_precip=None):
    '''
    Generate samples by reverse diffusion using guided_forward.
    '''
    model.eval()
    x = torch.randn(shape, device=device)  # Start from pure noise

    for t_ in reversed(range(num_steps)):
        t_batch = torch.full((x.shape[0],), t_, device=device, dtype=torch.long)
        x = p_sample(model, x, t_batch, schedules, static_attrs, x_cond, guidance_weight, future_precip)

    return x

@torch.no_grad()
def ddim_sample(
    model,
    shape,              # (batch_size, horizon)
    device,
    static_attrs=None,  # (batch_size, static_size) or None
    future_precip=None, # (batch_size, horizon, 1)
    eta=0.0,            # 0 for deterministic
    steps=50
):
    """
    DDIM reverse diffusion, velocity-prediction mode.
    model(x_past, t, static_attrs, future_precip) -> v_pred ([B,H])
    """
    B = shape[0]
    # uniform timesteps in [0,1]
    ts = torch.linspace(0., 1., steps, device=device)
    # start from pure noise
    x = torch.randn(shape, device=device)

    for i in range(steps - 1, -1, -1):
        t      = ts[i].repeat(B)                              # current t
        t_prev = ts[i-1].repeat(B) if i > 0 else ts[0].repeat(B)

        _, alpha_t,  sigma_t  = diffusion_params(t)
        _, alpha_tp, sigma_tp = diffusion_params(t_prev)

        # 1) network predicts velocity
        v = model(x, t, static_attrs, future_precip)          # [B,H]

        # 2) recover x0 and eps
        x0  = alpha_t * x - sigma_t * v
        eps = (v + sigma_t * x0) / alpha_t

        # 3) DDIM update
        if i > 0:
            sigma = eta * torch.sqrt(
                torch.clamp((sigma_tp**2) * (1 - (alpha_t**2)/(alpha_tp**2)), min=1e-12)
            )
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = alpha_tp * x0 + sigma_tp * eps + sigma * noise
        else:
            x = x0

    return x




