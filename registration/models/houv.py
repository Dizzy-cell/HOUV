import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch.autograd import Variable
from model_utils_completion import calc_cd_percent, loss_view
from train_utils import rotation_error, translation_error, rmse_loss

class HOUV(nn.Module):
    def __init__(self, batch_size, angle_base):
        super(HOUV, self).__init__()
        
        self.batch_size = batch_size
        self.angle_base = angle_base
        self.pi = torch.acos(torch.zeros(1)).item() * 2 

        vc_numpy = np.random.randn(batch_size, 3)
        num = 0
        for x0 in [-1,0,1]:
            for y0 in [-1,0,1]:
                for z0 in [-1, 0, 1]:
                    if x0 == 0 and y0 ==0 and z0 == 0:
                        continue
                    if num>=self.batch_size:
                        continue
                    vc_numpy[num] = np.array((x0, y0, z0))
                    num += 1

        self.V_c = nn.Parameter(data=torch.from_numpy(vc_numpy.astype(np.float32)), requires_grad = True) # 
        self.angle_c = nn.Parameter(data=torch.from_numpy(np.random.randn(batch_size, 1).astype(np.float32)), requires_grad = True) #[0,1] 
        self.tran_c = nn.Parameter(data=torch.from_numpy(np.random.randn(batch_size, 3).astype(np.float32)), requires_grad = True) #[0,0.5]
        self.tran_s_cpu = nn.Parameter(data=torch.from_numpy(np.random.randn(batch_size, 1).astype(np.float32)), requires_grad = True) #[0,1] 
        # self.tanh = nn.Tanh()
        self.batch_size = batch_size

    def reset_weight(self, batch_size, angle_base, seed=2021):
        self.batch_size = batch_size
        np.random.seed(seed)
        vc_numpy = np.random.randn(self.batch_size, 3)
        num = 0
        for x0 in [-1,0,1]:
            for y0 in [-1,0,1]:
                for z0 in [-1, 0, 1]:
                    if x0 == 0 and y0 ==0 and z0 == 0:
                        continue
                    vc_numpy[num] = np.array((x0, y0, z0))
                    num += 1

        self.angle_base = angle_base
        np.random.seed(seed)
        self.V_c = nn.Parameter(data=torch.from_numpy(vc_numpy.astype(np.float32)), requires_grad = True)# 
        np.random.seed(seed)
        self.angle_c = nn.Parameter(data=torch.from_numpy(np.random.randn(self.batch_size, 1).astype(np.float32)), requires_grad = True) #[0,1] 
        np.random.seed(seed)
        self.tran_c = nn.Parameter(data=torch.from_numpy(np.random.randn(self.batch_size, 3).astype(np.float32)), requires_grad = True) #[0,0.5]
        np.random.seed(seed)
        self.tran_s = nn.Parameter(data=torch.from_numpy(np.random.randn(self.batch_size, 1).astype(np.float32)), requires_grad = True) #[0,1] 

        A = torch.zeros((batch_size, 3, 3))
        self.A = A.cuda()

        ones = torch.zeros((batch_size, 3, 3)) + torch.eye(3)
        self.ones = ones.cuda()

    def cd_rotation(self, angle, V, device = 'cuda'):
        batch_size = angle.shape[0]
        V = V / torch.sqrt((V * V).sum(dim = 1, keepdim = True))

        #A = self.A

        A = torch.zeros((batch_size, 3, 3), dtype = torch.float)
        A = A.cuda()

        A[:, 0,1] = -V[:, 2]
        A[:, 0,2] = V[:, 1]
        A[:, 1,0] = V[:, 2]
        A[:, 1,2] = -V[:, 0]
        A[:, 2,0] = -V[:, 1]
        A[:, 2,1] = V[:, 0]

        R = self.ones + torch.sin(angle).unsqueeze(2) * A + (1 - torch.cos(angle)).unsqueeze(2) * torch.bmm(A,A)
        return R

    def translation(self, tran, s):
        tran = tran / torch.sqrt((tran * tran).sum(dim = 1, keepdim = True))
        tran = tran * s
        tran = tran.unsqueeze(1)
        return tran

    def forward(self, src):
        src = src.squeeze(0)
        angle = torch.sin(self.angle_c * self.pi) * self.pi / 8 + self.pi / 8 + self.angle_base * self.pi / 4 
        R = self.cd_rotation(angle, self.V_c)

        tran_s = torch.sin(self.tran_s * self.pi) * 0.125 + 0.125
        T = self.translation(self.tran_c, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T
        return src_temp, R, T


def predict_model(net, src, src_rotated, pose = None,  src_ori = None, tgt_ori=None, angle_t=None, label = None, kernel = 64, num_epochs = 500, angle_base=0, device='cuda', seed=2021):
    batch_size = src.shape[0]
    num_points = src.shape[1]
    
    # We get multiple optimization results according to the input, and find the best result.
    src = src.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, num_points, 3)
    src_rotated = src_rotated.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, num_points, 3)

    batch_size *= kernel

    net.reset_weight(batch_size, angle_base, seed=seed)
    net.cuda()
    optimizer = optim.Adam(net.parameters(), lr = 0.01)
    net.train()
    for i in range(num_epochs):
        optimizer.zero_grad()
        src_temp, R, T = net(src)
        rmse_loss, min_1 = Predict_loss(src_temp, src_rotated)
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

    batch_size = batch_size // kernel

    # src_temp, R, T = net(src)
    # _, min_1 = Predict_loss(src_temp, src_rotated)

    #rmse_loss = rmse_loss.reshape((batch_size, kernel)) # compare loss 
    rmse_loss = (min_1).reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))

    return rmse_loss, R, T



def solve_model(net, src, src_rotated, pose = None, src_ori = None, tgt_ori=None, angle_t=None, label=None, kernel = 64, num_epochs = 200, prefix = 'train'):
    batch_size = src.shape[0]
    R_lst = []
    rmse_lst = []
    T_lst = []
    rmse_min = []
    rmse_idx = []
    angle_num = 1
    rmse_mean = torch.zeros((1, batch_size))

    rmse_loss, R, T = predict_model(net, src, src_rotated, pose,  src_ori, tgt_ori, angle_t, label, kernel = kernel, num_epochs = num_epochs, angle_base= 0)
    rmse, b_index = rmse_loss.topk(1, dim = 1, largest = False, sorted = True)
    lst_add = []
    for j in range(batch_size):
        if rmse[j][0] > 0.030:
            lst_add.append(j)
    # print(label.sum(), len(lst_add))

    if len(lst_add) > 0:
        lst_add = np.array(lst_add)
        if pose == None:
            src_add, src_rotated_add, pose_add = src[lst_add], src_rotated[lst_add], None
        else:
            src_add, src_rotated_add, pose_add = src[lst_add], src_rotated[lst_add], pose[lst_add]
        lst_add = np.array(lst_add).astype(int)

        for base_angle in range(1,4):
            rmse_loss_add, R_add, T_add = predict_model(net, src_add, src_rotated_add, pose_add, kernel = kernel, num_epochs = num_epochs, angle_base= base_angle)

            rmse_add, b_index_add = rmse_loss_add.topk(1, dim = 1, largest = False, sorted = True)

            flag = (rmse_add < rmse[lst_add]).reshape(-1)
            flag = torch.nonzero(flag).reshape(-1)
            geid = lst_add[flag.int().cpu().numpy()]

            R[geid] = R_add[flag]
            rmse_loss[geid] = rmse_loss_add[flag]
            T[geid] = T_add[flag]
            rmse[geid] = rmse_add[flag]

    R_lst.append(R)
    rmse_lst.append(rmse_loss)
    T_lst.append(T)
    rmse_mean[0] = rmse_loss.min(dim = 1)[0].detach()
    
    ans = torch.zeros((batch_size, 4, 4))
    rmse_mean = rmse_mean.transpose(0,1).contiguous()
    rmse, idx = rmse_mean.topk(angle_num, dim = 1, largest = False, sorted = True)

    for i in range(batch_size):
        j = 0
        rm, id = rmse_lst[idx[i][j]][i].topk(1, dim = 0, largest = False, sorted = True)
        ans[i,:3,:3] = R_lst[idx[i][j]][i][id]
        ans[i,:3, 3] = T_lst[idx[i][j]][i][id]  

    ans = ans.cuda()

    if prefix == 'test':
        return ans.cpu()

    r_err = rotation_error(ans[:, :3, :3], pose[:, :3, :3])
    t_err = translation_error(ans[:,:3,3], pose[:,:3,3])

    print("Rotation error:", r_err.mean(), "Translation error:", t_err.mean())
    return r_err, t_err, ans


def Predict_loss(src, src_rotated, alpha = 0.5):
    # cd loss (Only consider the smallest 50%)
    cd_t, cd_p = calc_cd_percent(src, src_rotated, percent = alpha)
    min_1, _ = torch.min(torch.cat([cd_t.unsqueeze(1), cd_p.unsqueeze(1)], dim = 1), dim = 1)

    # cd loss projected onto three two-dimensional planes 
    cd_t_view_0, cd_p_view_0 = loss_view(src, src_rotated, dim = 0)
    cd_t_view_1, cd_p_view_1 = loss_view(src, src_rotated, dim = 1)
    cd_t_view_2, cd_p_view_2 = loss_view(src, src_rotated, dim = 2)  
    min_v_0, _ = torch.min(torch.cat([cd_t_view_0.unsqueeze(1), cd_p_view_0.unsqueeze(1)], dim=1), dim = 1)
    min_v_1, _ = torch.min(torch.cat([cd_t_view_1.unsqueeze(1), cd_p_view_1.unsqueeze(1)], dim=1), dim = 1)
    min_v_2, _ = torch.min(torch.cat([cd_t_view_2.unsqueeze(1), cd_p_view_2.unsqueeze(1)], dim=1), dim = 1)

    return (min_1) * 6 + (min_v_0 + min_v_1 + min_v_2), min_1
