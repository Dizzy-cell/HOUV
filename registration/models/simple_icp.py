import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
import math

from torch.autograd import Variable
from model_utils_completion import calc_cd, calc_cd_percent
from visu_utils import plot_grid_pcd


class cd_keba(nn.Module):
    def __init__(self, batch_size):
        super(cd_keba, self).__init__()
        self.V_c = nn.Parameter(data=torch.from_numpy(np.random.randn(int(batch_size), 3)), requires_grad = True)
        self.angle_c = nn.Parameter(data=torch.from_numpy(np.random.randn(int(batch_size), 1)), requires_grad = True)#[0,1] 
        self.tran_c =  nn.Parameter(data=torch.from_numpy(np.random.randn(int(batch_size), 3)), requires_grad = True) #[0,0.5]
        self.tran_s =  nn.Parameter(data=torch.from_numpy(np.random.randn(int(batch_size), 1)), requires_grad = True) #[0,0.5]
        self.pi = torch.acos(torch.zeros(1)).item() * 2 

        self.tanh = nn.Tanh()
        # self.batch_size = batch_size

    def cd_rotation(self, angle, V, device = 'cuda'):
        batch_size = angle.shape[0]
        V = V / torch.sqrt((V * V).sum(dim = 1, keepdim = True))

        A = Variable(torch.zeros((batch_size, 3, 3)))
        A = A.cuda()

        A[:, 0,1] = -V[:, 2]
        A[:, 0,2] = V[:, 1]
        A[:, 1,0] = V[:, 2]
        A[:, 1,2] = -V[:, 0]
        A[:, 2,0] = -V[:, 1]
        A[:, 2,1]= V[:, 0]

        ones = torch.zeros((batch_size, 3, 3)) + torch.eye(3)
        ones = ones.cuda()
        R = ones + torch.sin(angle).unsqueeze(2) * A + (1 - torch.cos(angle)).unsqueeze(2) * torch.bmm(A,A)
        return R

    def translation(self, tran, s):
        tran = tran / torch.sqrt((tran * tran).sum(dim = 1, keepdim = True))
        tran = tran * s
        tran = tran.unsqueeze(1)
        return tran

    def forward(self, src):
        src = src.squeeze(0)
        angle = torch.sigmoid(self.angle_c) * self.pi * 2
        R = self.cd_rotation(angle, self.V_c)

        tran_s = self.tanh(self.tran_s) * 0.25
        T = self.translation(self.tran_c, tran_s)
        src_temp = torch.bmm(src, R.transpose(1,2)) + T
        return src_temp, R, T
