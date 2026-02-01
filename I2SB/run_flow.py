import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging

# 导入自定义模块
from rf_model import UNet
from rf_sampling import RestoraFlowSampler

# 导入原有工具 (保持您项目路径结构)
from reconstru_visual import visua_and_save
from mask_obtain import mask_obtain
import deal_sst_util
from easydict import EasyDict



class FMRunner:
    def __init__(self, opt, log):
        self.opt = opt
        self.log = log
        self.device = opt.device
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
        # 初始化 UNet (Flow Matching Model)
        self.net = UNet(
            input_channels=1,     # SST 数据为单通道
            input_height=64,
            ch=64,
            output_channels=1,
            dropout=0.1
        ).to(self.device)
        
        # self.ema = ExponentialMovingAverage(self.net.named_parameters(), decay=opt.ema)
        # self.ema.to(self.device)
        
        # 初始化采样器
        self.sampler = RestoraFlowSampler(self.net, self.device, ode_steps=opt.ode_steps, correction_steps=opt.correction_steps)
        
        log.info("[Model] Initialized Restora-Flow UNet.")

    def train(self, opt, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        optimizer = optim.AdamW(self.net.parameters(), lr=opt.lr, weight_decay=opt.l2_norm)
        
        for epoch in range(opt.epoches):
            self.net.train()
            epoch_loss = 0.0
            
            for i, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()

                # x0 是目标图像 (Clean Data)
                # x1_deg 是退化图像 (仅用于可视化，不参与 Flow Matching 训练)
                x0 = batch_y[:, 6:7] 
                x1_deg = batch_x[:, 6:7]
                
                # --- Independent Flow Matching Loss (无 OT) ---
                # 1. 采样噪声 z
                z = torch.randn_like(x0)
                
                # 2. 采样时间 t
                t = torch.rand(x0.shape[0], device=self.device)
                t_reshaped = t.view(-1, 1, 1, 1)
                
                # 3. 构造插值 x_t (Straight Path)
                # 您的数据已经是 1对1 的，这里 z 和 x0 直接配对即可
                x_t = (1 - t_reshaped) * z + t_reshaped * x0
                
                # 4. 目标速度 v_target (指向目标)
                v_target = x0 - z
                
                # 5. 预测与 Loss
                v_pred = self.net(x_t, t)
                loss = torch.mean((v_pred - v_target) ** 2)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch} | Loss: {epoch_loss / len(train_loader):.6f}")
            self.validate(epoch, val_loader, opt)

                # 可视化
                # if epoch == opt.visual_epoch:
                #     pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                #     visua_and_save('Train', epoch, i, x1_orig, f'miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                #     visua_and_save('Train', epoch, i, x1_com, 'x1_com', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                #     visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                #     visua_and_save('Train', epoch, i, x0, 'ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                # if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                #     pred_x0 = self.compute_pred_x0(step, xt, pred, anomaly_mask=None, clip_denoise=opt.clip_denoise)
                #     visua_and_save('Train', epoch, i, x1_com, 'x1_com', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                #     visua_and_save('Train', epoch, i, pred_x0, 'recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)

            print(f"Epoch {epoch} | Loss: {epoch_loss / len(train_loader):.6f}")
            self.validate(epoch, val_loader, opt)

    def validate(self, epoch, val_loader, opt):
        # 使用 EMA 参数进行评估
        # self.ema.load_averaged(self.net) # 可选: 如果想用 EMA 权重
        self.net.eval()
        total_mse, total_rmse, total_mae, total_r2 = [], [], [], []
        test_loss = np.zeros(opt.epoches, dtype=float)
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(val_loader, desc="Validating")):
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                x1 = batch_x[:, 6:7] # 损坏图像
                x0 = batch_y[:, 6:7] # 目标图像
                
                # 获取掩码 (1=已知/保留, 0=未知/修复)
                # 请根据您的 mask_obtain 函数确认返回值: 这里假设 mask_obtain 返回的 1 是云/缺失区域，0 是有效区域
                # Restora-Flow 需要: 1=有效(保留), 0=缺失(生成)
                # 如果 mask_obtain("mask", ...) 返回的是损坏区域的 mask (即 1=损坏)，则需要反转
                mask_orig = mask_obtain("mask", opt.mask_type, opt.corrup_rate, mask_num=1, batch_size=x1.shape[0]).to(self.device)
                mask = (mask_orig == 0).float()
                # 使用 Restora-Flow 进行修复
                # 注意：传入的 degraded_img 应该是 x1 (包含噪声/遮挡的值)
                reconstructed_image = self.sampler.sample_denoising(x1, sigma=0.3)
                
                if opt.clip_denoise:
                    reconstructed_image.clamp_(-1., 1.)
                   # 可视化
                if epoch == opt.visual_epoch:
                    visua_and_save('Test_rf', epoch, i, x1, f'valid_miss{6}', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    visua_and_save('Test_rf', epoch, i, x0, 'valid_ground_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                if (epoch > opt.visual_epoch and epoch % opt.visual_epoch == 0):
                    visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)

                difference = reconstructed_image-x0
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
                        visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_min_rmse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mse >= mse_score.item():
                        self.mim_mse = mse_score.item()
                        self.mim_mse_epoch = epoch
                        self.mim_mse_batch = i + 1
                        visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_min_mse_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_mae >= mae_score.item():
                        self.mim_mae = mae_score.item()
                        self.mim_mae_epoch = epoch
                        self.mim_mae_batch = i + 1
                        visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_min_mae_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)
                    if self.mim_r2 <= R2_score.item():
                        self.mim_r2 = R2_score.item()
                        self.mim_r2_epoch = epoch
                        self.mim_r2_batch = i + 1
                        visua_and_save('Test_rf', epoch, i, reconstructed_image, 'valid_min_r2_recons', opt.N_S_ratio, opt.mask_type, opt.corrup_rate)

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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    opt = EasyDict({
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'mask_type': 'Cloud_mask',
        'corrup_rate': 8,
        'batch_size': 1, 
        'epoches': 200,
        'lr': 1e-4,
        'l2_norm': 0.0,
        # 'ema': 0.999,
        'clip_denoise': None,
        'visual_epoch': 5,
        'N_S_ratio': 0.1,
        'path': 'result_0.1_8.txt',
        'save_file': 'South_Sea',
        'ode_steps': 128,         # Flow Matching 采样步数
        'correction_steps': 1   # Restora-Flow 修正步数 (推荐=1)
    })

    # 加载数据 (保持不变)
    x_train, y_train, x_valid, y_valid = deal_sst_util.read_cache(
        f'./data/{opt.N_S_ratio}_{opt.mask_type}_{opt.corrup_rate}_train_{opt.save_file}_miss.h5'
    )
    
    train_data = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
    val_data = TensorDataset(torch.FloatTensor(x_valid), torch.FloatTensor(y_valid))

    runner = FMRunner(opt, log)
    runner.train(opt, train_data, val_data)