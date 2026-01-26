import numpy as np
from tqdm import tqdm
from functools import partial
import torch

from utils import unsqueeze_xdim
import torch.nn.functional as F
from ipdb import set_trace as debug

def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var
def gaussian_filter(x, sigma=1.0, kernel_size=3):
    """ Apply Gaussian filter to smooth values """
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding)

class Diffusion():
    def __init__(self, betas, device):
        self.device = device

        # compute analytic std: eq 11
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """
        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""
        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, log_steps=None, verbose=True):
        xt = x1.detach().to(self.device)
        xs = []
        pred_x0s = []
        log_steps = log_steps or steps
        assert steps[0] == log_steps[0] == 0
        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='DDPM sampling', total=len(steps)-1) if verbose else pair_steps
        for prev_step, step in pair_steps:
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)
            if mask is not None:
                xt_true = x1
                if not ot_ode:
                    _prev_step = torch.full((xt.shape[0],), prev_step, device=self.device, dtype=torch.long)
                    std_sb = unsqueeze_xdim(self.std_sb[_prev_step], xdim=x1.shape[1:])
                    xt_true = xt_true + std_sb * torch.randn_like(xt_true)
                xt = mask * xt_true + (1. - mask) * xt  # mask=1 (已有) 保留，mask=0 (缺失) 去噪
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach())
                xs.append(xt.detach())
        stack_bwd_traj = lambda z: torch.flip(torch.stack(z, dim=1), dims=(1,))
        return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)

class AsyncDiffusion(Diffusion):
    def __init__(self, betas, device, tau_min=0.2, tau_max=0.5):
        super().__init__(betas, device)
        self.tau_min = int(tau_min * len(betas))
        self.tau_max = int(tau_max * len(betas))
        self.total_steps = len(betas)

    def get_pixelwise_shifted_t(self, anomaly_mask, t):
        """计算像素级的偏移时间步，加入 Gaussian 平滑"""
        # anomaly_mask 已归一化到 [0, 1]，0 表示低异常值
        tau = self.tau_min + anomaly_mask * (self.tau_max - self.tau_min)
        # 应用 Gaussian 滤波平滑 tau
        tau_smoothed = gaussian_filter(tau, sigma=1.0, kernel_size=3)
        t_expanded = t.view(-1, 1, 1, 1).expand_as(anomaly_mask)
        shifted_t = torch.clamp(t_expanded - tau_smoothed.long(), 0, self.total_steps - 1)
        return shifted_t
    
    def get_std_fwd_pixel(self, step, anomaly_mask, xdim=None):
        """像素级 std_fwd，用于 compute_label 和 compute_pred_x0"""
        shifted_t = self.get_pixelwise_shifted_t(anomaly_mask, step)
        std_fwd = self.std_fwd[shifted_t]  # [B, 1, H, W]
        # print("s",std_fwd)
        return std_fwd
    
    def q_sample(self, step, x0, x1, anomaly_mask=None, ot_ode=False):
        """ Sample q(x_t | x_0, x_1) with shifted time steps """
        if anomaly_mask is None:
            return super().q_sample(step, x0, x1, ot_ode)

        # 计算像素级偏移时间步
        shifted_t = self.get_pixelwise_shifted_t(anomaly_mask, step)  # [B, H, W]

        # 使用 gather 提取每个像素对应的值
        mu_x0 = self.mu_x0[shifted_t]        # [B, H, W]
        mu_x1 = self.mu_x1[shifted_t]        # [B, H, W]
        std_sb = self.std_sb[shifted_t]      # [B, H, W]
        
        xt = mu_x0 * x0 + mu_x1 * x1
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0, anomaly_mask=None, ot_ode=False):
        """ Sample p(x_{nprev} | x_n, x_0) with shifted time steps """
        if anomaly_mask is None:
            return super().p_posterior(nprev, n, x_n, x0, ot_ode)

        shifted_nprev = self.get_pixelwise_shifted_t(anomaly_mask, nprev)
        shifted_n = self.get_pixelwise_shifted_t(anomaly_mask, n)

        std_n = self.std_fwd[shifted_n]        # [B, 1, H, W]
        std_nprev = self.std_fwd[shifted_nprev] # [B, 1, H, W]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)
        return xt_prev
    def async_ddpm_sampling(self, steps, pred_x0_fn, x1, anomaly_mask, ot_ode=False, log_steps=None, verbose=True):
        """异步采样，使用偏移时间步"""
        xt = x1.detach().to(self.device)
        xs = [xt.cpu()]
        pred_x0s = []
        log_steps = log_steps or [steps[-1]]
        assert steps[0] == 0
        steps = steps[::-1]
        pair_steps = zip(steps[1:], steps[:-1])
        pair_steps = tqdm(pair_steps, desc='AsyncDDPM', total=len(steps)-1) if verbose else pair_steps
        batch_size = xt.shape[0]
        for prev_step, step in pair_steps:
            step_tensor = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            prev_step_tensor = torch.full((batch_size,), prev_step, device=self.device, dtype=torch.long)
            pred_x0 = pred_x0_fn(xt, step)
            xt = self.p_posterior(prev_step_tensor, step_tensor, xt, pred_x0, anomaly_mask, ot_ode=ot_ode)
            if prev_step in log_steps:
                pred_x0s.append(pred_x0.detach().cpu())
                xs.append(xt.detach().cpu())
        result_xs, result_pred_x0s = torch.stack(xs, dim=1), torch.stack(pred_x0s, dim=1)
        del xs, pred_x0s
        torch.cuda.empty_cache()
        return result_xs[:, 1:], result_pred_x0s