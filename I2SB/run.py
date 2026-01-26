import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import datetime
from matplotlib import pyplot as plt
import logging
from torch.nn import functional as F
from reconstru_visual import visua_and_save
from mask_obtain import mask_obtain
import deal_sst_util
from easydict import EasyDict
from utils import *
from diffusion import Diffusion
from model import Image64Net, ExponentialMovingAverage

class DSBRunner:
    def __init__(self, opt, log):
        self.opt = opt
        self.log = log
        # 单批次最佳指标
        self.mim_rmse = float('inf')
        self.mim_mse = float('inf')
        self.mim_mae = float('inf')
        self.mim_r2 = -float('inf')
        self.mim_rmse_epoch = 0
        self.mim_mse_epoch = 0
        self.mim_mae_epoch = 0
        self.mim_r2_epoch = 0
        self.mim_rmse_batch = 0
        self.mim_mse_batch = 0
        self.mim_mae_batch = 0
        self.mim_r2_batch = 0
        
        # 平均最佳指标
        self.min_mean_rmse = float('inf')
        self.min_mean_mse = float('inf')
        self.min_mean_mae = float('inf')
        self.max_mean_r2 = -float('inf')
        self.min_mean_rmse_epoch = 0
        self.min_mean_mse_epoch = 0
        self.min_mean_mae_epoch = 0
        self.max_mean_r2_epoch = 0
         # 异步调度参数
        self.tau_min = opt.tau_min
        self.tau_max = opt.tau_max
        
        # 初始化扩散模型
        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        self.betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")
        
        # 初始化主网络
        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image64Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.net.to(opt.device)
        self.ema.to(opt.device)
      
        
    
    def compute_label(self, step, x0, xt, anomaly_mask=None):
        if anomaly_mask is not None:
            std_fwd = self.diffusion.get_std_fwd_pixel(step, anomaly_mask, xdim=x0.shape[1:])
        else:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
    
    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False, anomaly_mask=None):
        if anomaly_mask is not None:
            std_fwd = self.diffusion.get_std_fwd_pixel(step, anomaly_mask, xdim=xt.shape[1:])
        else:
            std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise:
            pred_x0.clamp_(-1., 1.)
        return pred_x0


    def train(self, opt, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        optimizer = optim.AdamW(self.net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
        mse = nn.MSELoss()
        
        # 训练指标记录
        train_loss = {'reconstruct': [], 'anomaly_constraint': []}
        epoch_avg_loss = {'reconstruct': np.zeros(opt.epoches), 'anomaly_constraint': np.zeros(opt.epoches)}
        
        for epoch in range(opt.epoches):
            self.net.train()
            epoch_reconstruct = 0.0
            
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader)):
                batch_x = batch_x.to(opt.device)
                batch_y = batch_y.to(opt.device)
                optimizer.zero_grad()

                # 获取数据
                x1 = batch_x[:, 6:7]  # 损坏图像
                x0 = batch_y[:, 6:7]  # 目标图像
                mask_orig = mask_obtain("mask", opt.mask_type, opt.corrup_rate, 
                                      mask_num=1, batch_size=batch_x.shape[0]).to(opt.device)
    
                step = torch.randint(0, opt.interval, (x0.shape[0],), device=opt.device, dtype=torch.long)
                xt = self.diffusion.q_sample(step, x0, x1)
                label = self.compute_label(step, x0, xt,).float()
                pred = self.net(xt, step, cond=x1).float()
                pred_x0 = self.compute_pred_x0(step, xt, pred, clip_denoise=opt.clip_denoise)
                # 损失项
                loss_true = mse(pred, label)
                # 总损失
                total_loss = loss_true 
                total_loss.backward()
                optimizer.step()
                self.ema.update()
                # 记录损失
                train_loss['reconstruct'].append(loss_true.item())
                epoch_reconstruct += loss_true.item()

                
                # 可视化
                if epoch == opt.visual_epoch:
                    pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                    visua_and_save('Train', epoch, i, x1, f'miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Train', epoch, i, x0, 'ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                    pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                    visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
            
            # 计算epoch平均损失
            epoch_avg_loss['reconstruct'][epoch] = epoch_reconstruct / len(train_loader)
            print(f'Train Epoch: {epoch} | Reconstruct_batch: {train_loss["reconstruct"][-1]:.6f} | Reconstruct_ave: {epoch_avg_loss["reconstruct"][epoch]:.6f}')
            
            # 绘制损失曲线
            # if epoch % opt.visual_epoch == 0:
            #     plt.figure(figsize=(12, 6))
            #     plt.plot(train_loss['reconstruct'], label='Reconstruct Loss (Batch)')
            #     plt.title('Training Loss (Batch Level)')
            #     plt.legend()
            #     plt.savefig(f'batch_loss_epoch{epoch}.png')
            #     plt.close()
                
            #     plt.figure(figsize=(12, 6))
            #     plt.plot(epoch_avg_loss['reconstruct'][:epoch+1], label='Reconstruct Loss (Epoch Avg)')
            #     plt.title('Training Loss (Epoch Average)')
            #     plt.legend()
            #     plt.savefig(f'epoch_loss_epoch{epoch}.png')
            #     plt.close()
            
            # 验证
            self.validate(epoch, val_loader, opt)

    def validate(self, epoch, val_loader, opt):
        self.net.eval()
        total_mse, total_rmse, total_mae, total_r2 = [], [], [], []
        test_loss = np.zeros(opt.epoches, dtype=float)
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(val_loader)):
                batch_x = batch_x.to(opt.device)
                batch_y = batch_y.to(opt.device)
                x1 = batch_x[:, 6:7].float()
                x0 = batch_y[:, 6:7].float()
                x3 = batch_y[:, 7:8].float()
                mask_orig = mask_obtain("mask", opt.mask_type, opt.corrup_rate, mask_num=1, batch_size=batch_x.shape[0]).to(opt.device)
                xs, pred_x0s = self.run_ddpm_sampling(opt,x1,cond=x1,clip_denoise=opt.clip_denoise,log_count=1, nfe=20)
                reconstructed_image = pred_x0s[:,-1]
                # print(" reconstructed_image",reconstructed_image.shape)

               
                # 可视化
                if epoch == opt.visual_epoch:
                    visua_and_save('Test', epoch, i, x1, f'valid_miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test', epoch, i, x0, 'valid_ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                    visua_and_save('Test', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                
                # 评估全图
                difference = reconstructed_image - x0
                mask = (mask_orig == 0).float()
                mse_score = torch.sum((difference * mask) ** 2) / torch.sum(mask)
                total_mse.append(mse_score.item())

                rmse_score = torch.sqrt(mse_score)
                total_rmse.append(rmse_score.item())

                mae_score = torch.sum(torch.abs(difference * mask)) / torch.sum(mask)
                total_mae.append(mae_score.item())

                SSE = torch.sum((difference ** 2) * mask)
                SST = torch.sum((reconstructed_image - (torch.sum(x0 * mask) / (torch.sum(mask))) * mask) ** 2)
                R2_score = 1 - SSE / SST
                total_r2.append(R2_score.item())
            
                if epoch >= opt.visual_epoch:
                    if self.mim_rmse >= rmse_score.item():
                        self.mim_rmse = rmse_score.item()
                        self.mim_rmse_epoch = epoch
                        self.mim_rmse_batch = i + 1
                        visua_and_save('Test_yb0', epoch, i, reconstructed_image, 'valid_min_rmse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mse >= mse_score.item():
                        self.mim_mse = mse_score.item()
                        self.mim_mse_epoch = epoch
                        self.mim_mse_batch = i + 1
                        visua_and_save('Test_yb0', epoch, i, reconstructed_image, 'valid_min_mse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mae >= mae_score.item():
                        self.mim_mae = mae_score.item()
                        self.mim_mae_epoch = epoch
                        self.mim_mae_batch = i + 1
                        visua_and_save('Test_yb0', epoch, i, reconstructed_image, 'valid_min_mae_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_r2 <= R2_score.item():
                        self.mim_r2 = R2_score.item()
                        self.mim_r2_epoch = epoch
                        self.mim_r2_batch = i + 1
                        visua_and_save('Test_yb0', epoch, i, reconstructed_image, 'valid_min_r2_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)

            print('Valid Epoch: {}\t RMSE Loss: {:.8f}\t MSE Loss: {:.8f} \t MAE Loss: {:.8f} \t R2 Loss: {:.8f}\t'
                  .format(epoch, np.mean(total_rmse), np.mean(total_mse), np.mean(total_mae), np.mean(total_r2)))

            # global min_mean_rmse, min_mean_mse, min_mean_mae, max_mean_r2
            # global min_mean_rmse_epoch, min_mean_mse_epoch, min_mean_mae_epoch, max_mean_r2_epoch
            if epoch >= opt.visual_epoch:
                if self.min_mean_rmse >= np.mean(total_rmse):
                    self.min_mean_rmse = np.mean(total_rmse)
                    self.min_mean_rmse_epoch = epoch
                if self.min_mean_mse >= np.mean(total_mse):
                    self.min_mean_mse = np.mean(total_mse)
                    self.min_mean_mse_epoch = epoch
                if self.min_mean_mae >= np.mean(total_mae):
                    self.min_mean_mae = np.mean(total_mae)
                    self.min_mean_mae_epoch = epoch
                if self.max_mean_r2 <= np.mean(total_r2):
                    self.max_mean_r2 = np.mean(total_r2)
                    self.max_mean_r2_epoch = epoch

            test_loss[epoch] = np.mean(total_mse)

        result = [self.min_mean_rmse, self.min_mean_rmse_epoch, self.min_mean_mse, self.min_mean_mse_epoch,
                 self.min_mean_mae, self.min_mean_mae_epoch, self.max_mean_r2, self.max_mean_r2_epoch]
        with open(opt.path, 'w') as file:
            file.write("best: Rmse:" + str(result[0]) + "[Epoch:" + str(result[1]) + "] "
                       + "mse:" + str(result[2]) + "[Epoch:" + str(result[3]) + "] "
                       + "mae:" + str(result[4]) + "[Epoch:" + str(result[5]) + "] "
                       + "r2:" + str(result[6]) + "[Epoch:" + str(result[7]) + "]")

        print("batch最低rmse:{:.8f}  [Epoch{}_batch{}]".format(self.mim_rmse, self.mim_rmse_epoch, self.mim_rmse_batch))
        print("batch最低mse:{:.8f}  [Epoch{}_batch{}]".format(self.mim_mse, self.mim_mse_epoch, self.mim_mse_batch))
        print("batch最低mae:{:.8f}  [Epoch{}_batch{}]".format(self.mim_mae, self.mim_mae_epoch, self.mim_mae_batch))
        print("batch最高r2:{:.8f}  [Epoch{}_batch{}]".format(self.mim_r2, self.mim_r2_epoch, self.mim_r2_batch))
        print('Min_mean_rmse:{:.8f} [Epoch{}]\t Min_mean_mse:{:.8f} [Epoch{}]\t Min_mean_mae:{:.8f} [Epoch{}]\t Max_mean_r2:{:.8f} [Epoch{}]\t'
              .format(self.min_mean_rmse, self.min_mean_rmse_epoch, self.min_mean_mse, self.min_mean_mse_epoch,
                     self.min_mean_mae, self.min_mean_mae_epoch, self.max_mean_r2, self.max_mean_r2_epoch))

        plt.figure(figsize=(12, 6))
        #plt.plot(train_loss['reconstruct'], label='Training Loss')
        plt.plot(test_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('train_valid_loss.png')
        plt.close()
                

    def run_ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):
        nfe = nfe or opt.interval - 1
        steps = space_indices(opt.interval, nfe + 1)
        log_count = min(len(steps) - 1, log_count)
        log_steps = [steps[i] for i in space_indices(len(steps) - 1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None:
            cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0s = self.diffusion.ddpm_sampling(
                steps=steps,
                pred_x0_fn=pred_x0_fn,
                x1=x1,
                mask=mask,
                ot_ode=opt.ot_ode,
                log_steps=log_steps,
                verbose=verbose
            )

        return xs.to(opt.device), pred_x0s.to(opt.device)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    opt = EasyDict({
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'use_fp16': False,
        'cond_x1': True,
        'mask_type': 'Cloud_mask',
        'corrup_rate': 8,
        'interval': 1000,
        'beta_max': 0.02,
        'batch_size': 1,
        'microbatch': 1,
        'epoches': 200,
        'lr': 1e-4,
        'l2_norm': 0.01,
        'ema': 0.999,
        'ot_ode': False,
        'clip_denoise': True,
        'visual_epoch': 5,
        'N_S_ratio': 0.1,
        'save_file': 'South_Sea',
        'path': 'result_8_0.1.txt',
        't0': 0.0,
        'T': 1.0,
        'tau_min': 0.2,
        'tau_max': 0.5,
        'verbose': True 
    })

    # 加载数据
    x_train, y_train, x_valid, y_valid = deal_sst_util.read_cache(
        f'./data/{opt.N_S_ratio}_{opt.mask_type}_{opt.corrup_rate}_train_{opt.save_file}_miss.h5'
    )
    
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))


    # 初始化Runner
    runner = DSBRunner(opt, log)
    runner.train(opt, train_data, val_data)
