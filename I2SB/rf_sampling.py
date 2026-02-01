import torch
import numpy as np
from tqdm import tqdm

# ==========================================
# Part 1: Scheduler (严格复现源码逻辑)
# ==========================================

def _check_times(times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)


def _plot_times(x, times):
    import matplotlib.pyplot as plt
    plt.plot(x, times)
    plt.show()


def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):

    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts


# ==========================================
# Part 2: RestoraFlow Sampler (包含去噪和补全)
# ==========================================

class RestoraFlowSampler:
    def __init__(self, model, device, ode_steps=64, correction_steps=1):
        self.model = model
        self.device = device
        self.steps_ode = int(ode_steps) 
        self.correction_steps = int(correction_steps)

    def sample_denoising(self, input_img, sigma=0.5):
        """
        [Source Code Port]: restora_flow.py -> sample_denoising
        适用于纯去噪任务 (SDEdit / Partial Diffusion 思路)。
        
        Args:
            input_img: 含噪的输入图像
            sigma: 噪声水平/重绘强度 (0.0 ~ 1.0)。
                   值越大，重绘越多（去噪越强但可能改变原结构）；
                   值越小，保留原图越多。
                   源码默认 logic 是 input_img * (1 - sigma)。
        """
        batch_size = input_img.shape[0]
        
        # 1. 初始化
        # 源码逻辑：从纯噪声开始，但前段部分强制替换为观测值
        x = torch.randn_like(input_img, device=self.device)
        
        # 构造观测值 x_obs (源码中做了缩放)
        # 注意：如果您的输入 input_img 已经是归一化好的数据，这里可能需要调整
        # 源码：x_obs = input_img * (1 - self.args.sigma_noise)
        x_obs = input_img * (1 - sigma)

        # 2. 时间步 (0 -> 1)
        # 源码直接用 linspace，没有用 jump schedule
        torch_linspace = torch.linspace(0, 1, int(self.steps_ode), device=self.device)
        delta_t = 1 / len(torch_linspace)

        for t in tqdm(torch_linspace, desc="Denoising"):
            mask = torch.ones(input_img.shape, device=self.device)

            if t < (1 - sigma):
                x = mask * x_obs + (1 - mask) * x
            else:
                # Flow Matching ODE Update
                x = x + delta_t * self.model(x, t.repeat(x.shape[0]))

        return x

    def sample_mask_based(self, input_img, mask, progress=False):
        """
        [Source Code Port]: restora_flow.py -> sample_mask_based
        适用于补全/Inpainting 任务 (Mask-Guided)。
        注意：此源码版本会强制保留 input_img 在 mask=1 区域的像素值。
        
        Args:
            input_img: 退化/含噪图像 (x1)
            mask: 1=已知区域(保留), 0=未知区域(补全)
        """
        batch_size = input_img.shape[0]
        x = torch.randn_like(input_img, device=self.device)
        pred_x_start = None
        
        if self.correction_steps < 1:
             raise ValueError("Number of correction steps must be >= 1.")

        # 1. 获取 Jump Schedule
        times = get_schedule_jump(
            t_T=self.steps_ode,
            n_sample=1,
            jump_length=1,
            jump_n_sample=self.correction_steps + 1
        )
        
        # 2. 归一化并反转 (0 -> 1)
        times_min, times_max = min(times), max(times)
        times = [((t - times_min) / (times_max - times_min)) for t in times]
        times.reverse()
        
        time_pairs = list(zip(times[:-1], times[1:]))
        if progress: 
            time_pairs = tqdm(time_pairs, desc="Inpainting")
            
        # 3. 采样循环
        for t_last, t_cur in time_pairs:
            t_last_tensor = torch.tensor([t_last] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            t_cur_tensor = torch.tensor([t_cur] * batch_size, device=self.device).view(batch_size, 1, 1, 1)
            
            # Forward (ODE Step)
            if t_last < t_cur:
                with torch.no_grad():
                    # --- Mask-Guided Fusion ---
                    if pred_x_start is not None:
                        # 构造 z_prim (模拟 t_last 时刻的数据)
                        eps = torch.randn_like(x)
                        z_prim = t_last_tensor * input_img + (1 - t_last_tensor) * eps
                        
                        # [源码核心]: 强制替换已知区域
                        x = mask * z_prim + (1 - mask) * x
                    
                    # --- Flow ODE Update ---
                    delta_t = (t_cur_tensor - t_last_tensor)
                    t_input = torch.tensor(t_last, device=self.device).repeat(batch_size)
                    v_pred = self.model(x, t_input)
                    x = x + delta_t * v_pred
                    
                    pred_x_start = True 
            
            # Backward (Correction Step)
            else:
                with torch.no_grad():
                    # --- Trajectory Correction ---
                    t_input = torch.tensor(t_last, device=self.device).repeat(batch_size)
                    v_pred = self.model(x, t_input)
                    x_1_prim = x + (1 - t_last_tensor) * v_pred
                    
                    x = t_cur_tensor * x_1_prim + (1 - t_cur_tensor) * torch.randn_like(x)

        return x