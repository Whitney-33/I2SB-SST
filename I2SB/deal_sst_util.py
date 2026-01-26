from netCDF4 import Dataset
import numpy as np
import os
import random
import torch
from sklearn.preprocessing import MaxAbsScaler,MinMaxScaler

import h5py


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")




def dealNaN(data):
    N=0
    for i in range(0,data.shape[0]):
        for j in range(0,data.shape[1]):
            if type(data[i][j])==np.ma.core.MaskedConstant:
                data[i][j]=0    #将空值置为0
                N+=1
    return data,N


# 数据集分割随机
def trainTestSplit_radom(trainingSet, targetSet, train_size):
    totalNum = int(len(trainingSet))     #训练集的总数，48400
    trainIndex = list(range(totalNum))  # 存放训练集的下标
    x_train = []  # 存放训练集输入
    y_train = []  # 存放训练集输出
    x_valid = []  # 存放验证集输入
    y_valid = []  # 存放验证集输出
    trainNum = int(totalNum * train_size)  # 划分训练集的样本数

    for i in range(trainNum):    #70%
        randomIndex = int(random.uniform(0, len(trainIndex)))
        x_train.append(trainingSet[randomIndex])
        y_train.append(targetSet[randomIndex])
        del (trainIndex[randomIndex])  # 删除已经放入训练集的下标
    for i in range(totalNum - trainNum):   #30%
        x_valid.append(trainingSet[trainIndex[i]])
        y_valid.append(targetSet[trainIndex[i]])
    return x_train, y_train, x_valid, y_valid


def trainTestSplit(trainingSet, targetSet, train_size):
    totalNum = int(len(trainingSet))
    trainIndex = list(range(totalNum))  # 存放训练集的下标
    x_train = []  # 存放训练集输入
    y_train = []  # 存放训练集输出
    x_valid = []  # 存放验证集输入
    y_valid = []  # 存放验证集输出
    if train_size == 'last_year':
        trainNum = totalNum - 1200
        print('totalNum', totalNum)
        print('trainNum', trainNum)
    else:
        trainNum = int(totalNum * train_size)  # 划分训练集的样本数
    for i in range(trainNum):
        x_train.append(trainingSet[i])
        y_train.append(targetSet[i])
    for j in range(trainNum, totalNum):
        x_valid.append(trainingSet[j])
        y_valid.append(targetSet[j])
        # print(np.array(x_train).shape)
    return x_train, y_train, x_valid, y_valid

# 数据集封装h5
def cache(fname, x_train, y_train, x_valid, y_valid):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('x_train', data=x_train)
    h5.create_dataset('y_train', data=y_train)
    h5.create_dataset('x_valid', data=x_valid)
    h5.create_dataset('y_valid', data=y_valid)

# 数据集封装h5
def cache_all(fname, data):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('data', data=data)

def read_cache(fname):
    f = h5py.File(fname, 'r')
    x_train, y_train, x_valid, y_valid = [], [], [], []
    x_train = f['x_train'][()]
    y_train = f['y_train'][()]
    x_valid = f['x_valid'][()]
    y_valid = f['y_valid'][()]
    return x_train, y_train, x_valid, y_valid

# 读取h5数据集
def read_cache_all(fname):
    f = h5py.File(fname, 'r')
    data=[]
    data= f['data'][()]
    return data

#读取NC文件
def read_nc(fname,dataname):
    f = Dataset(fname)
    data=f.variables[dataname][0]
    return data

# 数据归一化
def data_normal(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    mmn_data = scaler.transform(data)
    return mmn_data

# patch分割
def patch_split(datalist,winW,winH):     #[[[21],[21],[21],[21].....21]。。。。。。。共48400个]
    # print(datalist.shape[0],datalist.shape[1])
    h=datalist.shape[0]
    w=datalist.shape[1]
    new_data = []
    # 自定义滑动窗口的大小
    # 步长大小
    stepSize = 1                                             #datalist.shape=(240,240)
    for i in range(0,datalist.shape[0]-(h-winH*(h//winH)), winH):             #(h-winH*(h//winH))
        for j in range(0, datalist.shape[1]-(w-winW*(w//winW)), winW):         #(w-winW*(w//winW))
            # print(datalist[i:i+winW,j:j+winH])
            data = datalist[i:i + winW, j:j + winH]
            new_data.append(data)
    # print(len(new_data),"-------------")
    # print(np.array(new_data).shape)
    return np.array(new_data)

# 预处理
def get_data_h5(path,lat_start,lat_end,lon_start,lon_end):
    nc = read_cache_all(path)
    data = nc[lat_start: lat_end, lon_start: lon_end]
    return data

def get_data_nc(path,dataname,lat_start,lat_end,lon_start,lon_end):
    nc = read_nc(path,dataname)
    data = nc[lat_start: lat_end, lon_start: lon_end]
    return data





def split_sequence(sequence1,sequence2, sliding_window_width,sw_len):
    X, Y = [], []
    print('序列长度',len(sequence1))
    for i in range(len(sequence1)):
        # 找到最后一次滑动所截取数据中最后一个元素的索引，
        # 如果这个索引超过原序列中元素的索引则不截取；
        end_element_index1 = i + sliding_window_width
        end_element_index2=end_element_index1+sw_len
        # print(end_element_index)
        if end_element_index1 > len(sequence1): # 序列中最后一个元素的索引
            break
        sequence_x = sequence1[i:end_element_index1] # 取最后一个元素作为预测值y
        sequence_y= sequence2[i:end_element_index1]
        X.append(sequence_x)
        # print(len(sequence_x))
        Y.append(sequence_y)
    # print('X序列长度',np.array(X).shape)
    # return X,Y
    return np.array(X), np.array(Y)