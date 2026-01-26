import torch
from torch import nn
import time
import numpy as np
import torch.utils.data as Data
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import deal_sst_util

import numpy as np
import imageio
import matplotlib.ticker as ticker
import matplotlib.pyplot as pyplot
from PIL import Image
import cv2

def mask_obtain(type, mask_type, cor_rate,mask_num, batch_size):
    if type== "mask":
        mask = deal_sst_util.read_cache_all('./data/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
        mask = np.array(mask[0])

        if mask_num==1:
            return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0),dim=0)
        else:
            data = mask
            all_data = []
            for i in range (mask_num):
                all_data.append(data)
            # mask_ = np.array(all_data)
        final_data = []
        for i in range(batch_size):
            final_data.append(all_data)
        mask_ = np.array(final_data)
        return torch.tensor(mask_).float()

            # return torch.unsqueeze(torch.tensor(mask_).float(),dim=1)
    # elif type == "noise":
    #     mask = deal_sst_util.read_cache_all(
    #         '../data/AIN/data/{}_{:.0f}'.format(noise_type, cor_rate) + '.h5')  # mask 84 64 64
    #     mask = np.array(mask[0])
    #
    #     if mask_num == 1:
    #         return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0), dim=0)
    #     else:
    #         data = mask
    #         all_data = []
    #         for i in range(mask_num):
    #             all_data.append(data)
    #         mask_ = np.array(all_data)
    #
    #         return torch.unsqueeze(torch.tensor(mask_).float(), dim=1)
    # elif type == "fusion_mask":
    #     mask = deal_sst_util.read_cache_all(
    #         '../data/AIN/data/{}_{:.0f}'.format(noise_type, cor_rate) + '.h5')  # mask 84 64 64
    #     mask = np.array(mask[0])
    #     noise = deal_sst_util.read_cache_all('../data/AIN/data/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
    #     noise = np.array(noise[0])
    #
    #     fusion_mask  = mask + noise
    #     mask = fusion_mask > 1
    #     if mask_num == 1:
    #         return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0), dim=0)
    #     else:
    #         data = mask
    #         all_data = []
    #         for i in range(mask_num):
    #             all_data.append(data)
    #         mask_ = np.array(all_data)
    #
    #         return torch.unsqueeze(torch.tensor(mask_).float(), dim=0)
    # elif type == "fusion_mask_batch":
    #     mask = deal_sst_util.read_cache_all(
    #         '../dat_/AIN/data_/{}_{:.0f}'.format(noise_type, cor_rate) + '.h5')  # mask 84 64 64
    #     mask = np.array(mask[0])
    #     noise = deal_sst_util.read_cache_all('../data_/AIN/data_/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
    #     noise = np.array(noise[0])
    #
    #     fusion_mask  = mask + noise
    #     mask = fusion_mask > 1
    #     if mask_num == 1:
    #         return torch.unsqueeze(torch.unsqueeze(torch.tensor(mask).float(), dim=0), dim=0)
    #     else:
    #         data_ = mask
    #         all_data = []
    #         for i in range(mask_num):
    #             all_data.append(data_)
    #         # mask_ = np.array(all_data)
    #
    #     final_data = []
    #     for i in range(batch_size):
    #         final_data.append(all_data)
    #     mask_ = np.array(final_data)
    #     return torch.tensor(mask_).float()
    #
