import torch
from torch import nn
import time
import numpy as np
import torch.utils.data as Data
import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
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
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(torch.__version__)
print(torch.cuda.is_available())
print("using {} device.".format(device))
def create_file(train_type,batch_num, N_S_ratio,mask_type, cor_rate):
    if os.path.exists('./picture/{}_{}_{}'.format(N_S_ratio,mask_type, cor_rate)):
        if os.path.exists('./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate,train_type)):
            if os.path.exists('./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate,train_type, batch_num)):
                return 1
            else:
                os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate,train_type, batch_num))
        else:
            os.mkdir(r'./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate,train_type))
            os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate, train_type, batch_num))
    else:
        os.mkdir(r'./picture/{}_{}_{}'.format(N_S_ratio,mask_type, cor_rate))
        os.mkdir(r'./picture/{}_{}_{}/{}'.format(N_S_ratio,mask_type, cor_rate, train_type))
        os.mkdir(r'./picture/{}_{}_{}/{}/batch_{}'.format(N_S_ratio,mask_type, cor_rate, train_type, batch_num))



def visua_and_save(train_type,epoch,batch_num,data_input, path, N_S_ratio, mask_type, cor_rate):
    create_file(train_type,batch_num,N_S_ratio,mask_type, cor_rate)
    data = data_input.cpu().detach().numpy()

    # print("data.shape",data.shape)
    mask = deal_sst_util.read_cache_all('./data/{}_{:.0f}'.format(mask_type,cor_rate) + '.h5') #mask 84 64 64
    mask = np.array(mask[0])
    # for i in range(mask.shape[0]): #将黑变白，白变黑
    #     for j in range(mask.shape[1]):
    #         if mask[i][j] ==1:
    #             mask[i][j] = 0
    #         elif mask[i][j] ==0:
    #             mask[i][j] = 255
    # cv2.imwrite("./mask_image.png", mask)

    #
    # noise = deal_sst_util.read_cache_all('../data/AIN/data/{}_{:.0f}'.format(noise_type,cor_rate) + '.h5')
    # noise = np.array(noise[0])
    # for i in range(noise.shape[0]):  # 将黑变白，白变黑
    #     for j in range(noise.shape[1]):
    #         if noise[i][j] == 1:
    #             noise[i][j] = 0
    #         elif noise[i][j] == 0:
    #             noise[i][j] = 255
    # cv2.imwrite("./noise_image.png", noise)
    #
    # mask_image = cv2.imread("./mask_image.png")
    # noise_image = cv2.imread("./noise_image.png")
    # mask_ =  cv2.add(mask_image,noise_image)
    # cv2.imwrite("./mask_fusion.png", mask_)
    for i in range(data.shape[0]):

        y = torch.FloatTensor(data[i]).to(device)
        mask = torch.FloatTensor(data[i] != 0).to(device)

        # y = (max-min)*y + mean
        # y = (y -min)*35 / (max-min)
        #
        for j in range(y.shape[0]):
            # data2.shape(200, 200)
            if j== 7:
                filename= 'average'
            else:
                filename ='dayily'
            data2 = y[j]
            data2 = torch.squeeze(data2)
            data2 = torch.squeeze(data2).cpu().detach().numpy() #data2.shape torch.Size([200, 200])
            if path == "miss0" or path == "miss1" or path == "miss2" or path == "miss3"  or \
                    path == "miss4" or path == "miss5" or path == "miss6" or \
                path == "valid_miss0" or path == "valid_miss1" or path == "valid_miss2" or path == "valid_miss3" or  \
                    path == "valid_miss4" or path == "valid_miss5" or path == "valid_miss6"  :
                """画图"""
                mask = cv2.imread("./mask_image_{}.png".format(cor_rate))
                mask_ = cv2.resize(mask, dsize=(data2.shape[0], data2.shape[1]), dst=None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST)
                mask_ = mask_[:, :, 0]
                mask_ = mask_ > 220
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, mask=mask_, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30})  # , vmin=0, vmax=35
                # cbar = ax.collections[0].colorbar
                # cbar.ax.tick_params(labelsize=13)  # 调整颜色条刻度字体大小
                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',
                # fontsize  可选  normal/italic/oblique

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13)  # 如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)
                """
                lat_start = 2877  # 维度起点 -29.875 S
                lat_end = 2941  # 维度终点  -32.5416 S
                lon_start = 2517  # 经度起点  -75.125 W
                lon_end = 2581  # 经度终点  -72.458 W
                """
                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio, mask_type, cor_rate,train_type,batch_num)+str(epoch)  +'-'+ path +'.png', dpi=300)
                plt.clf()
                plt.cla()
                plt.close("all")

                # plt.show()
            elif path== "valid_min_rmse_recons" or path== "valid_min_mse_recons" or path== "valid_min_mae_recons" or path== "valid_min_r2_recons"  :
                """画图"""
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30})  # , vmin=0, vmax=35
                # cbar = ax.collections[0].colorbar
                # cbar.ax.tick_params(labelsize=13)  # 调整颜色条刻度字体大小
                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',
                # fontsize  可选  normal/italic/oblique

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13)  # 如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)
                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio , mask_type, cor_rate,train_type,batch_num)+ str(epoch)+ '-'+ path + '.png', dpi=300)
                plt.clf()
                plt.cla()
                plt.close("all")

                # plt.show()
            else:
                """画图"""
                plt.figure()
                ax = sns.heatmap(data2, cmap='jet', square=True, vmin=-1, vmax=1,
                                 annot_kws={"fontsize": 30})  # , vmin=0, vmax=35
                # cbar = ax.collections[0].colorbar
                # cbar.ax.tick_params(labelsize=13)  # 调整颜色条刻度字体大小
                plt.xlabel("°W", fontsize=30, style="normal", labelpad=-5.0, rotation=0, x=1.11)  # fontweight ='bold',
                plt.ylabel("°N", fontsize=30, style="normal", labelpad=-33.0, rotation=1, y=1.01)  # fontweight ='bold',
                # fontsize  可选  normal/italic/oblique

                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                name_list = ('70', '69.5', '69', '68.5', '68', '67.5', '67')
                plt.xticks(np.arange(0, 65, 64 / 6), name_list,
                           rotation=0, fontsize=13)  # 如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
                name_list = ('26', '25.5', '25', '24.5', '24', '23.5', '23')
                plt.yticks(np.arange(0, 65, 64 / 6), name_list, fontsize=13)
                plt.savefig('./picture/{}_{}_{}/{}/batch_{}/'.format(N_S_ratio, mask_type, cor_rate,train_type,batch_num)+str(epoch) +'-'+ path + '.png', dpi=300)

                plt.clf()
                plt.cla()
                plt.close("all")
                # plt.show()
                """
                lat_start = 2877  # 维度起点 -29.875 S
                lat_end = 2941  # 维度终点  -32.5416 S
                lon_start = 2517  # 经度起点  -75.125 W
                lon_end = 2581  # 经度终点  -72.458 W
                """
    plt.close()
    # elif path =="ground_recons" or path =="recons":
    #     for i in range(data.shape[0]):
    #         y = torch.FloatTensor(data[i]).to(device)
    #         mask = torch.FloatTensor(data[i] > 0).to(device)
    #
    #         y = (max-min)*y + mean
    #         y = (y -min)*35 / (max-min)
    #         # min = 271.34999999999997
    #         # max = 309.34999999999997
    #         # y = ((y - min) / (max - min))
    #         # y = y* 35
    #         # y = y * mask
    #
    #         for j in range(y.shape[0]):
    #             # data2.shape(200, 200)
    #             if j == 4:
    #                 filename = 'average'
    #             else:
    #                 filename = 'dayily'
    #             data2 = y[j]
    #             data2 = torch.squeeze(data2)
    #             data2 = torch.squeeze(data2).cpu().detach().numpy()  # data2.shape torch.Size([200, 200])
    #
    #         if path =="ground_recons" or path =="recons":
    #             plt.figure()
    #             ax = sns.heatmap(data2, cmap='jet', square=True,vmin=0,vmax=35)#, vmin=0,vmax=35, vmin=0,vmax=1
    #             plt.xlabel("°W",fontsize =20,style = "normal",labelpad=-5.0, rotation=0, x=1.09)#fontweight ='bold',
    #             plt.ylabel("°S",fontsize =20,style = "normal",labelpad=-33.0, rotation=1, y=1.01) # fontweight ='bold',
    #             #fontsize  可选  normal/italic/oblique
    #
    #             ax.spines['top'].set_visible(True)
    #             ax.spines['right'].set_visible(True)
    #             ax.spines['left'].set_visible(True)
    #             ax.spines['bottom'].set_visible(True)
    #             name_list = ('-75','-74.5' ,'-74', '-73.5', '-73', '-72.5','-72')
    #             plt.xticks(np.arange(0,65,64/6), name_list,rotation=0) #如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
    #             name_list = ('-29','-29.5','-30' ,  '-30.5', '-31', '-31.5','-32')
    #             plt.yticks(np.arange(0, 65, 64/ 6), name_list)
    #             plt.savefig('./picture/{}_{}_{}/'.format(mask_type,noise_type,cor_rate)+str(epoch) +'-'+str(batch_num) +'-'+ path +'-'
    #                        + str(i)+ filename + str(j)+'.png', dpi=300, bbox_inches='tight')
    #             # plt.show()
    #             """
    #             lat_start = 2877  # 维度起点 -29.875 S
    #             lat_end = 2941  # 维度终点  -32.5416 S
    #             lon_start = 2517  # 经度起点  -75.125 W
    #             lon_end = 2581  # 经度终点  -72.458 W
    #             """
    #             # print('保存完成...')
    #
    # else:
    #     for i in range(data.shape[0]):
    #         y = torch.FloatTensor(data[i]).to(device)
    #         mask = torch.FloatTensor(data[i] > 0).to(device)
    #         y = (max-min)*y + mean
    #         y = (y -min)*35 / (max-min)
    #         # min = 271.34999999999997
    #         # max = 309.34999999999997
    #         # y = ((y - min) / (max - min))
    #         # y = y* 35
    #         # y = y * mask
    #
    #         for j in range(y.shape[0]):
    #             # data2.shape(200, 200)
    #             if j == 4:
    #                 filename = 'average'
    #             else:
    #                 filename = 'dayily'
    #             data2 = y[j]
    #             data2 = torch.squeeze(data2)
    #             data2 = torch.squeeze(data2).cpu().detach().numpy()  # data2.shape torch.Size([200, 200])
    #
    #         if path =="valid_ground" or path =="valid_recons":
    #             plt.figure()
    #             ax = sns.heatmap(data2, cmap='jet', square=True,vmin=0,vmax=35)#, vmin=0,vmax=35, vmin=0,vmax=1
    #             plt.xlabel("°W",fontsize =20,style = "normal",labelpad=-5.0, rotation=0, x=1.09)#fontweight ='bold',
    #             plt.ylabel("°S",fontsize =20,style = "normal",labelpad=-33.0, rotation=1, y=1.01) # fontweight ='bold',
    #             #fontsize  可选  normal/italic/oblique
    #
    #             ax.spines['top'].set_visible(True)
    #             ax.spines['right'].set_visible(True)
    #             ax.spines['left'].set_visible(True)
    #             ax.spines['bottom'].set_visible(True)
    #             name_list = ('-75','-74.5' ,'-74', '-73.5', '-73', '-72.5','-72')
    #             plt.xticks(np.arange(0,65,64/6), name_list,rotation=0) #如果是np.arange(0,64,64/6),以为最后一个格的索引是63，所以最后一个刻度显示不出
    #             name_list = ('-29','-29.5','-30' ,  '-30.5', '-31', '-31.5','-32')
    #             plt.yticks(np.arange(0, 65, 64/ 6), name_list)
    #             plt.savefig('./picture/{}_{}_{}/'.format(mask_type,noise_type,cor_rate)+str(epoch) +'-'+str(batch_num) +'-'+ path +'-'
    #                        + str(i)+ filename + str(j)+'.png', dpi=300, bbox_inches='tight')
    #             # plt.show()
    # plt.close()
