# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:33:41 2022

@author: Po-Yen Tung; Ziyuan Rao
"""
import cv2
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# A simple neural network classifier that predicts INVAR based on composition.
# 定义了一个名为“Classifier”的类，它继承了PyTorch中的nn.Module类。在该类的构造函数中，定义了一个名为“fc”的神经网络，它包含两个线性层和一个Sigmoid函数层。该网络的输入大小为2，输出大小为1。
class Classifier(nn.Module): #a very simple classifer with large dropout. intuition here: as simple as possible, given that we only have 2d input
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2,8),
            nn.Dropout(0.5),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fc(x)

#%%Classifier training, Again you can play around with params just to see how it affects the model accuracy (training and tesing)
# 设置了一些参数，包括分类器的批量大小（batch size）、学习率、训练轮数、交叉验证的折数等。通过调用上面定义的“Classifier”类，创建了一个名为“cls”的分类器实例，并定义了优化器“opt”（使用Adam优化器）。
from matplotlib.pyplot import MultipleLocator
same_seeds(1)

params['cls_bs'] = 16
params['cls_lr'] = 1e-4
params['cls_epoch'] = 100
params['num_fold'] = 5


params['label_y'] = np.where(raw_y<5, 1, 0)
params['latents'] = latents

cls = Classifier().to(device) #创建类的实例并将其移动到指定的（CPU 或 GPU）进行计算
opt = Adam(cls.parameters(), lr=params['cls_lr'], weight_decay=0.) #创建一个 Adam 优化器，其学习率在字典中指定，权重衰减为 0.0。优化器用于在训练期间更新神经网络的参数


def training_Cls(model, optimizer, params): #定义一个将神经网络模型、优化器和参数字典作为输入的函数
    label_y = params['label_y'] #从字典中获取 INVAR 标签
    latents = params['latents'] #从字典中获取潜在向量
    cls_epoch = params['cls_epoch'] #从字典中获取训练分类器的周期数

    kf = KFold(n_splits=params['num_fold']) #使用字典中指定的折叠数创建 KFold 交叉验证器
    train_acc = [] #创建空列表以存储每个折叠的训练和测试精度
    test_acc = []

    k=1 #初始化计数器变量以跟踪当前折叠
    for train, test in kf.split(latents): #循环遍历每个折叠的训练和测试索引
        x_train, x_test, y_train, y_test = latents[train], latents[test], label_y[train], label_y[test] #将潜在向量和INVAR标签拆分为当前折叠的训练集和测试集。
        cls_dataset = AttributeDataset(x_train, y_train) #从训练数据创建对象。
        cls_dataloader = DataLoader(cls_dataset, batch_size=params['cls_bs'], shuffle=True) #使用字典中指定的批量大小为 创建对象，并在训练期间随机播放样本。
        cls_testDataset = AttributeDataset(x_test, y_test) #从测试数据创建对象。
        cls_testDataloader = DataLoader(cls_testDataset, batch_size=cls_testDataset.__len__(), shuffle=False) #为 创建一个对象，其批大小等于测试集中的样本数，并且在测试过程中不要随机播放样本。


        for epoch in range(cls_epoch): #循环遍历 epoch 数以训练分类器
            t = time.time() #获取当前时间。
            total_loss = []
            total_acc = [] #创建空列表以存储训练期间每个批次的损失和准确性
            cls.train() #将模型设置为训练模式。
            
            for i, data in enumerate(cls_dataloader): #循环训练批次。
                x = data[0].to(device) #获取输入数据并将其移动到指定的device
                y = data[1].to(device) #获取目标数据并将其移动到指定的device
                y_pred = cls(x) #通过模型馈送输入数据，以获得预测的 INVAR 概率
                loss = F.binary_cross_entropy(y_pred, y) #通过模型馈送输入数据，以获得预测的 INVAR 概率
                total_acc.append(torch.sum(torch.where(y_pred>=0.5,1,0) == y).detach().cpu().numpy()) #计算预测准确率，并将其存储在列表 total_acc 中。
                total_loss.append(loss.item()) #将本次迭代的损失值添加到列表 total_loss 中。

                opt.zero_grad() #将优化器opt中的梯度缓存清零
                loss.backward() #计算损失函数对模型参数的梯度
                opt.step() #更新模型参数
            
            #eval
            cls.eval() #将模型 cls 切换为评估模式
            for test in cls_testDataloader:
                x = test[0].to(device)
                y = test[1].to(device)
                y_pred = cls(x)
                accuracy = torch.sum(torch.where(y_pred>=0.5,1,0) == y) / y_pred.size(0)
                test_loss = F.binary_cross_entropy(y_pred, y)

            #print(f'[{epoch+1:03}/{cls_epoch}] loss:{sum(total_loss)/len(total_loss):.3f} test_loss:{test_loss.item():.3f} acc:{sum(total_acc)/cls_dataset.__len__():.3f} test_acc:{accuracy:.3f} time:{time.time()-t:.3f}')
        
        print('[{}/{}] train_acc: {:.04f} || test_acc: {:.04f}'.format(k, params['num_fold'], sum(total_acc)/cls_dataset.__len__(), accuracy.item()))
        train_acc.append(sum(total_acc)/cls_dataset.__len__())
        test_acc.append(accuracy.item())
        k+=1
    print('train_acc: {:.04f} || test_acc: {:.04f}'.format(sum(train_acc)/len(train_acc), sum(test_acc)/len(test_acc)))
    plt.figure()
    sns.set_style()
    plt.xlabel('number of folds')
    plt.ylabel('loss')
    x=range(1,params['num_fold']+1)
    sns.set_style("darkgrid")
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    plt.plot(x, train_acc)
    plt.plot(x, test_acc, linestyle=':', c='steelblue')
    plt.legend(["train_accuracy", "test_accuracy"])
    ax.xaxis.set_major_locator(x_major_locator)
    plt.savefig('figure/binary_classifier.png',dpi=300)
    return train_acc, test_acc

train_acc, test_acc = training_Cls(cls, opt, params)
