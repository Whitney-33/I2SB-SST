import torch
import torch.nn as nn
from guided_diffusion.script_util import create_model
from utils import count_parameters

class Image64Net(nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=True):
        super().__init__()
        kwargs = {
            "image_size": 64,
            "num_channels": 128,
            "num_res_blocks": 2,
            "channel_mult": "",
            "learn_sigma": False,
            "class_cond": False,
            "use_checkpoint": False,
            "attention_resolutions": "16,8",
            "num_heads": 4,
            "num_head_channels": 64,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": True,
            "dropout": 0.0,
            "resblock_updown": True,
            "use_fp16": use_fp16,
            "use_new_attention_order": False,
            "in_channels": 2 if cond else 1,
            "out_channels": 1,
            "dtype": torch.float32  # 显式指定模型参数类型
        }
        self.diffusion_model = create_model(**kwargs).float()  # 强制转换为float32
        log.info(f"[Net] Initialized network for 64x64x1! Size={count_parameters(self.diffusion_model)}")
        self.cond = cond
        self.noise_levels = noise_levels.float()  # 确保噪声级别是float32

    def forward(self, x, steps, cond=None):
        # 强制输入为float32
        x = x.float()
        if cond is not None:
            cond = cond.float()
            
        t = self.noise_levels[steps].detach()
        assert t.dim() == 1 and t.shape[0] == x.shape[0]
        x = torch.cat([x, cond], dim=1) if self.cond else x
        return self.diffusion_model(x, t)

# ExponentialMovingAverage 类保持不变
class ExponentialMovingAverage:
    def __init__(self, parameters, decay):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for param in self.parameters:
            self.shadow[param] = param.data.clone()

    def update(self):
        for param in self.parameters:
            self.shadow[param] = (1. - self.decay) * param.data + self.decay * self.shadow[param]

    def average_parameters(self):
        class Context:
            def __init__(self, ema):
                self.ema = ema
            def __enter__(self):
                for param in self.ema.parameters:
                    self.ema.backup[param] = param.data
                    param.data = self.ema.shadow[param]
            def __exit__(self, *args):
                for param in self.ema.parameters:
                    param.data = self.ema.backup[param]
        return Context(self)

    def to(self, device):
        for param in self.shadow:
            self.shadow[param] = self.shadow[param].to(device)
        return self