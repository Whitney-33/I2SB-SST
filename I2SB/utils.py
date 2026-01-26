import os
import datetime
import numpy as np
import torch
from functools import partial
from torch_ema import ExponentialMovingAverage

def space_indices(num_steps, count):
    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)
    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride
    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = (torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2).numpy()
    return betas

def create_file():
    now_time = datetime.datetime.now().replace(microsecond=0)
    now_time = now_time.strftime("%Y_%m_%d_%H_%M")
    dir = './save_model/' + now_time
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir