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

EPOCH = 201
BATCH_SIZE = 64
LR = 0.0001
drug_num = 1307
protein_num = 1996
indication_num = 3926
drug_feature = 1024  # ECFPs指纹
a1 = 0.0000000001
a2 = 0.001

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
            nn.Linear(2048, 1996),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1996, 4096),
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


if __name__ ==  '__main__':
    test_drug_fps = np.loadtxt('dataset/test_fps.txt')
    test_x = torch.from_numpy(test_drug_fps).float()

    model = torch.load("result/model.pkl")
    model.eval()
    preDTI, preRDA = model(test_x)
    np.savetxt('result/y_pre_DPI.txt', preDTI.detach().numpy(), fmt='%f')
