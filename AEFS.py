import torch
import torch.nn as nn
# import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from my_function import *
from sklearn.metrics import auc
from torch.utils.data import TensorDataset, DataLoader
import copy

EPOCH = 200
BATCH_SIZE = 64
LR = 0.0001
drug_num = 1307
protein_num = 1996
indication_num = 3926
drug_feature = 1024  # ECFPs指纹
a1 = 0.00000001
a2 = 0.0001


class AutoEncoder(nn.Module):
    def __init__(self, size_x, size_y):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(size_x, 2048),
            nn.Dropout(0.2),         # 参数是扔掉的比例
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 2048),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, protein_num),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(protein_num, 4096),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.BatchNorm1d(4096),
            nn.Linear(4096, size_y),
            nn.Dropout(0.2),
        )

    # def forward(self, x, sp):
    def forward(self, x):
        e0 = self.encoder(x)
        e1 = F.softmax(e0, dim=1)
        d0 = self.decoder(e1)
        d1 = F.softmax(d0, dim=1)
        return e1, d1


class AEFSLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, e, th, sr, sp, a):  # 定义前向的函数运算即可
        l2_1 = torch.mm(e, e.T)
        l2_2 = torch.mm(torch.sqrt_(torch.sum(e.mul(e), dim=1).view(torch.sum(e.mul(e), dim=1).shape[0], 1)), torch.sqrt_(torch.sum(e.mul(e), dim=1).view(1, torch.sum(e.mul(e), dim=1).shape[0])))
        l2_3 = torch.div(l2_1, l2_2) - sr
        l2 = torch.sum(l2_3.mul(l2_3)) / (e.shape[0] * e.shape[0])
        l3_1 = torch.mm(e.T, e)
        l3_2 = torch.mm(torch.sqrt_(torch.sum(e.T.mul(e.T), dim=1).view(torch.sum(e.T.mul(e.T), dim=1).shape[0], 1)), torch.sqrt_(torch.sum(e.T.mul(e.T), dim=1).view(1, torch.sum(e.T.mul(e.T), dim=1).shape[0])))
        l3_3 = l3_1 / l3_2 - sp
        l3 = torch.sum(l3_3.mul(l3_3)) / (e.shape[1] * e.shape[1])
        return a * l2 + a * l3


if __name__ == '__main__':
    print("读取数据")
    train_drug_idx = np.loadtxt('dataset/train_idx.txt')
    train_drug_indications = np.loadtxt('dataset/train_RDA.txt')
    train_drug_targets = np.loadtxt('dataset/train_DPI.txt')
    train_drug_fps = np.loadtxt('dataset/train_fps.txt')
    SP = np.loadtxt('dataset/SP.txt')
    SR = np.loadtxt('dataset/SR.txt')
    SD = np.loadtxt('dataset/SD.txt')
    print("处理数据")
    SP = max_min_normalize(SP)
    SR = max_min_normalize(SR)
    SD = max_min_normalize(SD)
    print("numpy 转 tensor")
    train_id = torch.from_numpy(train_drug_idx).int()
    train_x = torch.from_numpy(train_drug_fps).float()
    train_h = torch.from_numpy(train_drug_targets).float()
    train_y = torch.from_numpy(train_drug_indications).float()
    SP = torch.from_numpy(SP).float()
    SR = torch.from_numpy(SR).float()
    SD = torch.from_numpy(SD).float()
    print("初始化模型")
    train_set = TensorDataset(train_id, train_x, train_h, train_y)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    AE = AutoEncoder(drug_feature, indication_num)
    optimizer = torch.optim.Adam(AE.parameters(), lr=LR)
    loss_func = AEFSLoss()
    loss_func1 = nn.MSELoss()
    print("cuda加速")
    AE = AE.cuda()
    SP = SP.cuda()
    SD = SD.cuda()
    print("开始训练")
    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            batch_id, batch_x, batch_h, batch_y = data
            batch_SR = torch.empty(batch_x.shape[0], batch_x.shape[0])
            for m in range(batch_x.shape[0]):
                for n in range(batch_x.shape[0]):
                    batch_SR[m, n] = SR[batch_id[m], batch_id[n]]
            batch_SR = batch_SR.cuda()
            batch_x = batch_x.cuda()
            batch_h = batch_h.cuda()
            batch_y = batch_y.cuda()

            encoded, decoded = AE(batch_x)
            loss1 = loss_func1(encoded, batch_h) + loss_func(encoded, batch_h, batch_SR, SP, a1)
            loss2 = loss_func1(decoded, batch_y) + loss_func(decoded, batch_y, batch_SR, SD, a2)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:', epoch, 'train loss: %.20f' % loss.cpu().data)
    torch.save(AE.cpu(), 'result/model.pkl')
