#from MVP_Benchmark.MVP_Benchmark.registration.model_utils_completion import calc_cd
#from __future__ import print_function
#from _typeshed import OpenTextModeUpdating
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
from model_utils_completion import calc_cd, calc_cd_percent, calc_cd_percent_len, generate_sent_masks, loss_view
from visu_utils import plot_grid_pcd
from models.simple_icp import cd_keba

from mm3d_pn2 import furthest_point_sample, gather_points


class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(path, net):
    torch.save({'net_state_dict': net.module.state_dict()}, path)


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def rt_to_transformation(R, t):
	bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R.shape[0], 1, 1).to(R.device)
	T = torch.cat([torch.cat([R, t], dim=2), bot_row], dim=1)
	return T


def rotation_error(R, R_gt):
	cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
	cos_theta = torch.clamp(cos_theta, -1, 1)
	return torch.acos(cos_theta) * 180 / math.pi


def translation_error(t, t_gt):
	return torch.norm(t - t_gt, dim=1)


def rmse_loss(pts, T, T_gt):
	pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
	pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
	return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)


def rotation_geodesic_error(m1, m2):
	batch=m1.shape[0]
	m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3

	cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
	cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
	cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )

	theta = torch.acos(cos)

	#theta = torch.min(theta, 2*np.pi - theta)

	return theta


def rotation(angle, V, device = 'cuda'):
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

def rotation_v2(angle_xyz, device='cuda'):
    batch_size = angle_xyz.shape[0]
    angle_x, angle_y, angle_z = angle_xyz[:,0:1], angle_xyz[:,1:2],angle_xyz[:, 2:3]

    axis_x = torch.tensor([1, 0, 0]).expand(batch_size, 3).float().to(device)
    R_x = rotation(angle_x, axis_x)
    axis_y = torch.tensor([0, 1, 0]).expand(batch_size, 3).float().to(device)
    R_y = rotation(angle_y, axis_y)
    axis_z = torch.tensor([0, 0, 1]).expand(batch_size, 3).float().to(device)
    R_z = rotation(angle_z, axis_z)
    return torch.bmm(torch.bmm(R_x, R_y), R_z)

def translation(tran, s):
    tran = tran / torch.sqrt((tran * tran).sum(dim = 1, keepdim = True))
    tran = tran * s
    tran = tran.unsqueeze(1)
    return tran

def getPredict(src, src_rotated, pose, num_epochs = 1000):
    batch_size = src.shape[0]
    V_c = Variable(torch.ones(batch_size, 3), requires_grad = True) # 
    angle_c = Variable(torch.zeros((batch_size, 1)) + 0.5, requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.zeros((batch_size , 1, 3)) + 0.25, requires_grad = True) #[0,0.5]

    vari_list = [V_c, angle_c, tran_c]
    optimizer = optim.Adam(vari_list, lr = 0.01)

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.cuda()
        tran = tran_c.cuda()
        angle = angle_c.cuda()

        R = rotation(angle, V)
        T = translation(tran)

        src_temp = torch.bmm(src, R) + T 
        rmse_loss = torch.norm(src_temp - src_rotated, dim=2).mean(dim=1).mean()

        rmse_loss.backward()
        optimizer.step()


    r_err = rotation_error(R, pose[:,:3,:3].transpose(1,2))
    t_err = translation_error(T.squeeze(1), pose[:,:3,3])
    print(rmse_loss.mean(), r_err.mean(), t_err.mean())

    return 

def getPredict_cd(src, src_rotated, pose,  src_ori, tgt_ori, num_epochs = 1000):
    batch_size = src.shape[0]
    V_c = Variable(torch.ones(batch_size, 3), requires_grad = True) # 
    angle_c = Variable(torch.zeros((batch_size, 1)) + 0.5, requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.zeros((batch_size , 1, 3)) + 0.25, requires_grad = True) #[0,0.5]

    vari_list = [V_c, angle_c, tran_c]
    optimizer = optim.Adam(vari_list, lr = 0.01)

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.cuda()
        tran = tran_c.cuda()
        angle = angle_c.cuda()

        angle = torch.sigmoid(angle)
        tran = 1 * torch.tanh(tran)

        R = rotation(angle, V)
        T = translation(tran)

        src_temp = torch.bmm(src, R) + T 
        #rmse_loss = torch.norm(src_temp - src_rotated, dim=2).mean(dim=1).mean()


        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        rmse_loss = cd_t + cd_p
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()


    r_err = rotation_error(R, pose[:,:3,:3].transpose(1,2))
    t_err = translation_error(T.squeeze(1), pose[:,:3,3])
   #print(rmse_loss.mean(), r_err.mean(), t_err.mean())
    print(rmse_loss)
    print(r_err)
    print(t_err)
    src = src.detach().cpu().numpy()
    src_temp = src_temp.detach().cpu().numpy()
    src_rotated = src_rotated.detach().cpu().numpy()

    src_ori =  src_ori.detach().cpu().numpy()
    tgt_ori =  tgt_ori.detach().cpu().numpy()

    for i in range(src.shape[0]):
        plot_grid_pcd([src_ori[i], tgt_ori[i], src[i], src_temp[i], src_rotated[i]], shape = [1,5], save_path = 'Grads/{}.png'.format(i))

    from IPython import embed
    embed()
    
    return 



# kernel batch 
def getPredict_cd_keba(src, src_rotated, pose,  src_ori, tgt_ori, angle_t, num_epochs = 1000):
    batch_size = src.shape[0]
    V_c = Variable(torch.from_numpy(np.random.randn(batch_size, 3)), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]

    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        angle = torch.sigmoid(angle) * pi
        tran_s = torch.sigmoid(tran_s) * 0.5

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 

        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        rmse_loss = cd_t + cd_p
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()


    r_err = rotation_error(R, pose[:,:3,:3])
    t_err = translation_error(T.squeeze(1), pose[:,:3,3])
    rmse, idx = rmse_loss.topk(2, largest = False, sorted=True)
    
    print(rmse_loss[idx[0]], r_err[idx[0]], t_err[idx[0]]) 
    return r_err[idx[0]], t_err[idx[0]]


# kernel batch 
def getPredict_cd_keba_v2(src, src_rotated, pose,  src_ori, tgt_ori, angle_t, label, kernel = 32, num_epochs = 1000):
    batch_size = src.shape[0]
    
    src = src.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    src_rotated = src_rotated.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    #pose = pose.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 4, 4)
    #label = label.unsqueeze(1).expand((-1, kernel)).reshape((-1,))

    batch_size *= kernel

    V_c = Variable(torch.from_numpy(np.random.randn(batch_size, 3)), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]

    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        angle = torch.sigmoid(angle) * pi * 2
        tran_s = torch.sigmoid(tran_s) * 0.25

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 

        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        rmse_loss = cd_t + cd_p
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

    batch_size = batch_size // kernel
    rmse_loss = rmse_loss.reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))

    rmse, idx = rmse_loss.topk(1, dim = 1, largest = False, sorted = True)
    ans = torch.zeros((batch_size, 4, 4))

    for i in range(batch_size):
        ans[i, :3, :3] = R[i, idx[i,0]]
        ans[i, :3, 3] = T[i, idx[i, 0]]
        ans[i, 3, 3] = 1

    ans = ans.cuda()

    r_err = rotation_error(ans[:, :3, :3], pose[:, :3, :3])
    t_err = translation_error(ans[:,:3,3], pose[:,:3,3])

    r_err_sort, idx_s = r_err.sort()
    
    for j, err in enumerate(r_err):
        if err > 70:
            plot_grid_pcd([src[j * kernel + idx[j,0]].detach().cpu().numpy(), src_temp[j * kernel + idx[j,0]].detach().cpu().numpy(), src_rotated[j * kernel + idx[j,0]].detach().cpu().numpy()], shape = [1,3], save_path ='Grads/{}.png'.format(j * kernel + idx[j,0]), title = str(err))
            print('Grads/{}.png'.format(j * kernel + idx[j,0]))


    return r_err, t_err, ans




# kernel batch 
def getPredict_angle(src, src_rotated, pose = None,  src_ori = None, tgt_ori=None, angle_t=None, label = None, kernel = 64, num_epochs = 1000, angle_base = 0):
    batch_size = src.shape[0]
    num_points = src.shape[1]

    src = src.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, num_points, 3)
    src_rotated = src_rotated.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, num_points, 3)
    #pose = pose.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 4, 4)
    #label = label.unsqueeze(1).expand((-1, kernel)).reshape((-1,))

    batch_size *= kernel

    vc_numpy = np.random.randn(batch_size, 3)
    num = 0
    for x0 in [-1,0,1]:
        for y0 in [-1,0,1]:
            for z0 in [-1, 0, 1]:
                if x0 == 0 and y0 ==0 and z0 == 0:
                    continue
                vc_numpy[num] = np.array((x0, y0, z0))
                num += 1


    V_c = Variable(torch.from_numpy(vc_numpy), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]
    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    
    angle_XYZ_c = Variable(torch.from_numpy(np.random.randn(batch_size, 3)), requires_grad = True)

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu, angle_XYZ_c]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()
        angle_XYZ = angle_XYZ_c.float().cuda()


        angle = torch.sin(angle * pi) * pi / 8 + pi / 8 + angle_base * pi / 4
        tran_s = torch.sin(tran_s * pi) * 0.125 + 0.125 # MVP data
        #tran_s = torch.sin(tran_s * pi) * 0.25 + 0.25  # Other data
        #tran_s = torch.tanh(tran_s) * 0.125 + 0.125
        #trans_s = torch.sigmoid(tran_s) * 0.250

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 

        combine_data = src_rotated

        cd_t, cd_p = calc_cd_percent(src_temp, combine_data, percent = 0.5)
        # cd_t_percent_3_1, cd_p_percent_3_1 = calc_cd_percent(src_temp, combine_data, percent = 1) 
        #cd_t_percent_1, cd_p_percent_1 = calc_cd_percent(src_temp, combine_data, percent = 0.1) 

        # cd_t_view_0, cd_p_view_0 = loss_view(src_temp, combine_data, dim = 0)
        # cd_t_view_1, cd_p_view_1 = loss_view(src_temp, combine_data, dim = 1)
        # cd_t_view_2, cd_p_view_2 = loss_view(src_temp, combine_data, dim = 2)       

        min_1, _ = torch.min(torch.cat([cd_t.unsqueeze(1), cd_p.unsqueeze(1)], dim = 1), dim = 1)
        # min_3_1, _= torch.min(torch.cat([cd_t_percent_3_1.unsqueeze(1), cd_p_percent_3_1.unsqueeze(1)], dim = 1), dim = 1)
        # min_1_1,_ = torch.min(torch.cat([cd_t_percent_1.unsqueeze(1), cd_p_percent_1.unsqueeze(1)], dim=1), dim = 1)

        # min_v_0,_ = torch.min(torch.cat([cd_t_view_0.unsqueeze(1), cd_p_view_0.unsqueeze(1)], dim=1), dim = 1)
        # min_v_1,_ = torch.min(torch.cat([cd_t_view_1.unsqueeze(1), cd_p_view_1.unsqueeze(1)], dim=1), dim = 1)
        # min_v_2,_ = torch.min(torch.cat([cd_t_view_2.unsqueeze(1), cd_p_view_2.unsqueeze(1)], dim=1), dim = 1)

        #alpha = (min_1 / min_3_1).detach() * 8
        #beta = (min_1 / min_1_1).detach() * 24

        #rmse_loss = min_1 * 8 + min_v_0 * u + min_v_1 * v + min_v_2 * w
        #rmse_loss = min_1 
        
        rmse_loss =  (min_1) * 6  #+ (min_v_0 + min_v_1 + min_v_2)
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

        # print("Angle:", angle[:10] * 180 / pi)
        # print("Tran", tran_s[:10])



    batch_size = batch_size // kernel

    #rmse_loss = rmse_loss.reshape((batch_size, kernel)) # compare loss 
    rmse_loss = (min_1).reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))

    return rmse_loss, R, T, tran_s


def combine(src, tgt):
    data = torch.cat([src, tgt], dim = 1).transpose(1,2).contiguous()

    sample_idx = furthest_point_sample(data, 2048)
    combine = gather_points(data, sample_idx)
    return combine.transpose(1,2).contiguous()


def solve(src, src_rotated, pose = None,  src_ori = None, tgt_ori=None, angle_t=None, label=None, kernel = 64, num_epochs = 500, prefix = 'train'):
    batch_size = src.shape[0]

    R_lst = []
    rmse_lst = []
    T_lst = []
    rmse_min = []
    rmse_idx = []

    angle_num = 1

    rmse_mean = torch.zeros((angle_num, batch_size))

    # pt_id  = furthest_point_sample(src, 1024)
    # src = gather_points(src.transpose(1,2).contiguous(), pt_id).transpose(1,2).contiguous()
    # pt_id = furthest_point_sample(src_rotated, 1024)
    # src_rotated =  gather_points(src_rotated.transpose(1,2).contiguous(), pt_id).transpose(1,2).contiguous()

    #print(src.shape, src_rotated.shape)

    for i in range(angle_num):
        rmse_loss, R, T, tran_s =  getPredict_angle(src, src_rotated, pose,  src_ori, tgt_ori, angle_t, label, kernel = kernel, num_epochs = 500, angle_base= i)

        rmse, b_index = rmse_loss.topk(1, dim = 1, largest = False, sorted = True)
        lst_add = []
        for j in range(batch_size):
           if rmse[j][0] > 0.030:
               lst_add.append(j)
        #print(label.sum(), len(lst_add))

        if len(lst_add) >= 1:
            src_add, src_rotated_add, pose_add = src[lst_add], src_rotated[lst_add], None
            lst_add = np.array(lst_add).astype(int)

            for base_angle in range(1,4):
                rmse_loss_add, R_add, T_add, trans_s_add = getPredict_angle(src_add, src_rotated_add, pose_add, kernel = kernel, num_epochs = 500, angle_base= base_angle)

                rmse_add, b_index_add = rmse_loss_add.topk(1, dim = 1, largest = False, sorted = True)

                flag = (rmse_add < rmse[lst_add]).reshape(-1)
                flag = torch.nonzero(flag).reshape(-1)
                geid = lst_add[flag.int().cpu().numpy()]

                R[geid] = R_add[flag]
                rmse_loss[geid] = rmse_loss_add[flag]
                T[geid] = T_add[flag]
                rmse[geid] = rmse_add[flag]
                
            # print((rmse[lst_add] > 0.030).sum())
            # from IPython import embed
            # embed()



 

        R_lst.append(R)
        rmse_lst.append(rmse_loss)
        T_lst.append(T)
        #rmse_mean[i] = rmse_loss.mean(dim = 1).detach()



        # tran_s_np = tran_s.reshape(batch_size, 32).detach().cpu().numpy()
        #print(tran_s)
        # a = ((tran_s != 0.250)).reshape(batch_size, 32).int() * (-1)
        # rmse_loss += a
        rmse_mean[i] = rmse_loss.min(dim = 1)[0].detach()
    
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

    print(r_err.mean(), t_err.mean(), rmse.max())

    for j, err in enumerate(r_err):
        if err > 70:

            src_temp = src[j] @ ans[j,:3,:3].T + ans[j,:3,3]
            plot_grid_pcd([src[j].detach().cpu().numpy(), src_temp.detach().cpu().numpy(),src_rotated[j].detach().cpu().numpy(), src_ori[j].detach().cpu().numpy(), tgt_ori[j].detach().cpu().numpy()], shape = [1,5], save_path ='Grads/newbad_without_{}.png'.format(j * kernel + idx[j,0]), title = str(err))
            print('Grads/new_{}.png'.format(j * kernel + idx[j,0]))

            # for i in range(4):    
            #     temp_r_err = rotation_error(R_lst[i][j], pose[j,:3,:3].expand((kernel, -1, -1)))
            #     print(temp_r_err)

            # print(j, label[j], angle_t[j], translation_back(pose[j,:3,:3].cpu().numpy()), translation_back(ans[j,:3,:3].detach().cpu().numpy()))
            # print(rmse[j])
            # from IPython import embed 
            # embed()

    return r_err, t_err, ans

# kernel batch 
def getPredict_cd_keba_v3(src, src_rotated, pose,  src_ori, tgt_ori, angle_t, label, kernel = 32, num_epochs = 1000):
    batch_size = src.shape[0]

    src = src.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    src_rotated = src_rotated.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    #pose = pose.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 4, 4)
    #label = label.unsqueeze(1).expand((-1, kernel)).reshape((-1,))

    batch_size *= kernel

    vc_numpy = np.random.randn(batch_size, 3)
    num = 0
    for x0 in [-1,0,1]:
        for y0 in [-1,0,1]:
            for z0 in [-1, 0, 1]:
                if x0 == 0 and y0 ==0 and z0 == 0:
                    continue
                vc_numpy[num] = np.array((x0, y0, z0))
                num += 1


    V_c = Variable(torch.from_numpy(vc_numpy), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]
    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    V_c2 = Variable(torch.from_numpy(vc_numpy), requires_grad = True) # 
    angle_c2 = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c2 =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]
    tran_s_cpu2 = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 


    vari_list = [V_c, angle_c, tran_c, tran_s_cpu, V_c2, angle_c2, tran_c2, tran_s_cpu2]
    optimizer = optim.Adam(vari_list, lr = 0.01)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        V2 = V_c2.float().cuda()
        tran2 = tran_c2.float().cuda()
        angle2 = angle_c2.float().cuda()
        tran_s2 = tran_s_cpu2.float().cuda()

        angle = torch.sigmoid(angle) * pi * 0.25
        tran_s = torch.sigmoid(tran_s) * 0.20

        angle2 = torch.sigmoid(angle2) * pi * 0.25 + pi * 0.25
        tran_s2 = torch.sigmoid(tran_s2) * 0.20


        R = rotation(angle, V)
        T = translation(tran, tran_s)

        R2 = rotation(angle2, V2)
        T2 = translation(tran2, tran_s2)
        

        src_temp = torch.bmm(src, R.transpose(1,2)) + T 
        src_temp2 = torch.bmm(src, R2.transpose(1,2)) + T2 

        #src_temp = torch.bmm(src, R.transpose(1,2))

        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated, percent = 1)
        cd_t_percent_3_1, cd_p_percent_3_1 = calc_cd_percent(src_temp, src_rotated, percent = 0.3) 
        cd_t_percent_1, cd_p_percent_1 = calc_cd_percent(src_temp, src_rotated, percent = 0.1) 

        cd_t_view_0, cd_p_view_0 = loss_view(src_temp, src_rotated, dim = 0)
        cd_t_view_1, cd_p_view_1 = loss_view(src_temp, src_rotated, dim = 1)
        cd_t_view_2, cd_p_view_2 = loss_view(src_temp, src_rotated, dim = 2)       

        min_1, _ = torch.min(torch.cat([cd_t.unsqueeze(1), cd_p.unsqueeze(1)], dim = 1), dim = 1)
        min_3_1, _= torch.min(torch.cat([cd_t_percent_3_1.unsqueeze(1), cd_p_percent_3_1.unsqueeze(1)], dim = 1), dim = 1)
        min_1_1,_ = torch.min(torch.cat([cd_t_percent_1.unsqueeze(1), cd_p_percent_1.unsqueeze(1)], dim=1), dim = 1)

        min_v_0,_ = torch.min(torch.cat([cd_t_view_0.unsqueeze(1), cd_p_view_0.unsqueeze(1)], dim=1), dim = 1)
        min_v_1,_ = torch.min(torch.cat([cd_t_view_1.unsqueeze(1), cd_p_view_1.unsqueeze(1)], dim=1), dim = 1)
        min_v_2,_ = torch.min(torch.cat([cd_t_view_2.unsqueeze(1), cd_p_view_2.unsqueeze(1)], dim=1), dim = 1)

        alpha = (min_1 / min_3_1).detach() * 8
        beta = (min_1 / min_1_1).detach() * 24
        u = (min_1 / min_v_0).detach() * 2
        v = (min_1 / min_v_1).detach() * 2
        w = (min_1 / min_v_2).detach() * 2


        rmse_loss = min_1 + min_3_1 * alpha + min_1_1 * beta + min_v_0 * u + min_v_1 * v + min_v_2 * w

        cd_t_2, cd_p_2 = calc_cd_percent(src_temp2, src_rotated, percent=1)
        cd_t_percent_3_2, cd_p_percent_3_2 = calc_cd_percent(src_temp2, src_rotated, percent = 0.3) 
        cd_t_percent_2, cd_p_percent_2 = calc_cd_percent(src_temp2, src_rotated, percent = 0.1) 

        min_2, _ = torch.min(torch.cat([cd_t_2.unsqueeze(1), cd_p_2.unsqueeze(1)], dim = 1), dim = 1)
        min_3_2, _ = torch.min(torch.cat([cd_t_percent_3_2.unsqueeze(1), cd_p_percent_3_2.unsqueeze(1)], dim = 1), dim = 1)
        min_1_2, _ = torch.min(torch.cat([cd_t_percent_2.unsqueeze(1), cd_p_percent_2.unsqueeze(1)], dim=1), dim = 1)

        cd_t_view_0, cd_p_view_0 = loss_view(src_temp2, src_rotated, dim = 0)
        cd_t_view_1, cd_p_view_1 = loss_view(src_temp2, src_rotated, dim = 1)
        cd_t_view_2, cd_p_view_2 = loss_view(src_temp2, src_rotated, dim = 2) 

        min_v_0,_ = torch.min(torch.cat([cd_t_view_0.unsqueeze(1), cd_p_view_0.unsqueeze(1)], dim=1), dim = 1)
        min_v_1,_ = torch.min(torch.cat([cd_t_view_1.unsqueeze(1), cd_p_view_1.unsqueeze(1)], dim=1), dim = 1)
        min_v_2,_ = torch.min(torch.cat([cd_t_view_2.unsqueeze(1), cd_p_view_2.unsqueeze(1)], dim=1), dim = 1)

        alpha_2 = (min_2 / min_3_2).detach() * 8
        beta_2 = (min_2 / min_1_2).detach() * 24
        u = (min_2 / min_v_0).detach() * 2
        v = (min_2 / min_v_1).detach() * 2
        w = (min_2 / min_v_2).detach() * 2

        
        rmse_loss2 = min_2 + min_3_2 * alpha_2 + min_1_2 * beta_2 + min_v_0 * u + min_v_1 * v + min_v_2 * w

        rmse = rmse_loss.mean() + rmse_loss2.mean()
        rmse.backward()
        optimizer.step()



    batch_size = batch_size // kernel

    #rmse_loss = rmse_loss.reshape((batch_size, kernel)) # compare loss 
    rmse_loss = min_1_1.reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))


    rmse_loss2 = min_1_2.reshape((batch_size, kernel))
    R2 = R2.reshape((batch_size, kernel, 3, 3))
    T2 = T2.reshape((batch_size, kernel, 3))

    rmse, idx = rmse_loss.topk(1, dim = 1, largest = False, sorted = True)
    rmse2, idx2 = rmse_loss2.topk(1, dim = 1, largest = False, sorted = True)

    ans = torch.zeros((batch_size, 4, 4))

    for i in range(batch_size):
        if rmse[i] < rmse2[i] * 1.3:
            ans[i, :3, :3] = R[i, idx[i,0]]
            ans[i, :3, 3] = T[i, idx[i, 0]]
            ans[i, 3, 3] = 1
        else:
            ans[i, :3, :3] = R2[i, idx2[i,0]]
            ans[i, :3, 3] = T2[i, idx2[i, 0]]
            ans[i, 3, 3] = 1
            
            src_temp[i * kernel: i * kernel+kernel] = src_temp2[i * kernel: i * kernel+kernel] # visulation
                                                              # loss

    ans = ans.cuda()

    r_err = rotation_error(ans[:, :3, :3], pose[:, :3, :3])
    t_err = translation_error(ans[:,:3,3], pose[:,:3,3])


    # r_err_sort, idx_s = r_err.sort()
    for j, err in enumerate(r_err):
        if err > 70:
            plot_grid_pcd([src[j * kernel + idx[j,0]].detach().cpu().numpy(), src_temp[j * kernel + idx[j,0]].detach().cpu().numpy(), src_rotated[j * kernel + idx[j,0]].detach().cpu().numpy(), src_ori[j].cpu().numpy(), tgt_ori[j].cpu().numpy()], shape = [1,5], save_path ='Grads/{}.png'.format(j * kernel + idx[j,0]), title = str(err))
            print('Grads/{}.png'.format(j * kernel + idx[j,0]))

            r_err_j = rotation_error(R[j], pose[j,:3,:3].expand((kernel, -1,-1)))
            print(r_err_j)

            r_err_j = rotation_error(R2[j], pose[j,:3,:3].expand((kernel, -1,-1)))
            print(r_err_j)

            print(rmse[j], rmse2[j])
            # from IPython import embed 
            # embed()



    print(r_err)
    # print(rmse)

    return r_err, t_err, ans


 #kernel batch 

def getPredict_cd_vox(src, src_rotated, src_len, tgt_len, pose, kernel = 32, num_epochs = 1000):
    batch_size = src.shape[0]


    src = src.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    src_rotated = src_rotated.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 2048, 3)
    src_len = src_len.unsqueeze(1).expand((-1, kernel)).reshape(-1)
    tgt_len = tgt_len.unsqueeze(1).expand((-1, kernel)).reshape(-1)

    #pose = pose.unsqueeze(1).expand((-1, kernel, -1,-1)).reshape(-1, 4, 4)
    #label = label.unsqueeze(1).expand((-1, kernel)).reshape((-1,))

    batch_size *= kernel

    vc_numpy = np.random.randn(batch_size, 3)
    num = 0
    for x0 in [-1,0,1]:
        for y0 in [-1,0,1]:
            for z0 in [-1, 0, 1]:
                if x0 == 0 and y0 ==0 and z0 == 0:
                    continue
                vc_numpy[num] = np.array((x0, y0, z0))
                num += 1


    V_c = Variable(torch.from_numpy(vc_numpy), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]

    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    mask1 = generate_sent_masks(src_len.shape[0], 2048, src_len.cpu().numpy()).detach().cuda()
    mask2 = generate_sent_masks(tgt_len.shape[0], 2048, tgt_len.cpu().numpy()).detach().cuda()

    # mask1 = None
    # mask2 = None

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        angle = torch.sigmoid(angle / 10) * pi / 4
        tran_s = torch.sigmoid(tran_s) * 0.25

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 
        #src_temp = torch.bmm(src, R.transpose(1,2))

        cd_t, cd_p = calc_cd_percent_len(src_temp, src_rotated, mask1, mask2)
        cd_t_percent, cd_p_percent = calc_cd_percent_len(src_temp, src_rotated, mask1, mask2, percent = 0.3) 
        cd_t_percent_1, cd_p_percent_1 = calc_cd_percent_len(src_temp, src_rotated, mask1, mask2, percent = 0.1) 


        rmse_loss = cd_t +  cd_p + (cd_t_percent + cd_p_percent) * 2 + (cd_t_percent_1 + cd_p_percent_1) * 3
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

        # print(rmse)

    batch_size = batch_size // kernel
    rmse_loss = rmse_loss.reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))

    rmse, idx = rmse_loss.topk(1, dim = 1, largest = False, sorted = True)
    ans = torch.zeros((batch_size, 4, 4))

    for i in range(batch_size):
        ans[i, :3, :3] = R[i, idx[i,0]]
        ans[i, :3, 3] = T[i, idx[i, 0]]
        ans[i, 3, 3] = 1

    ans = ans.cuda()

    r_err = rotation_error(ans[:, :3, :3], pose[:, :3, :3])
    t_err = translation_error(ans[:,:3,3], pose[:,:3,3])

    # r_err_sort, idx_s = r_err.sort()
    
    for j, err in enumerate(r_err):
        if err > 70:
            plot_grid_pcd([src[j * kernel + idx[j,0]].detach().cpu().numpy(), src_temp[j * kernel + idx[j,0]].detach().cpu().numpy(), src_rotated[j * kernel + idx[j,0]].detach().cpu().numpy()], shape = [1,3], save_path ='Grads/{}.png'.format(j * kernel + idx[j,0]), title = str(err))
            print('Grads/{}.png'.format(j * kernel + idx[j,0]))

            # from IPython import embed
            # embed()

    print(r_err)

    return r_err, t_err, ans

def getPredict_test_keba(src, src_rotated,  num_epochs = 1000):
    batch_size = src.shape[0]
    V_c = Variable(torch.from_numpy(np.random.randn(batch_size, 3)), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]

    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        angle = torch.sigmoid(angle) * pi
        tran_s = torch.sigmoid(tran_s) * 0.5

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 

        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        rmse_loss = cd_t + cd_p
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

    rmse, idx = rmse_loss.topk(2, largest = False, sorted=True)
    
    ans = torch.zeros((1, 4, 4))
    ans[:, :3, :3] = R[idx[0]]
    ans[:,:3, 3] = T[idx[0]] 
    ans[:,3, 3] = 1
    return ans 


def getPredict_test_keba_v2(src, src_rotated,  kernel = 32, num_epochs = 1000):
    batch_size = src.shape[0]

    src = src.unsqueeze(1).expand((-1,kernel,-1,-1)).reshape(-1,2048,3).cuda()
    src_rotated =  src_rotated.unsqueeze(1).expand((-1,kernel,-1,-1)).reshape(-1, 2048, 3).cuda()

    batch_size *= kernel

    vc_numpy = np.random.randn(batch_size, 3)
    num = 0
    for x0 in [-1,0,1]:
        for y0 in [-1,0,1]:
            for z0 in [-1, 0, 1]:
                if x0 == 0 and y0 ==0 and z0 == 0:
                    continue
                vc_numpy[num] = np.array((x0, y0, z0))
                num += 1

    V_c = Variable(torch.from_numpy(vc_numpy), requires_grad = True) # 
    angle_c = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 
    tran_c =  Variable(torch.from_numpy(np.random.randn(batch_size , 3)), requires_grad = True) #[0,0.5]

    tran_s_cpu = Variable(torch.from_numpy(np.random.randn(batch_size, 1)), requires_grad = True) #[0,1] 

    vari_list = [V_c, angle_c, tran_c, tran_s_cpu]
    optimizer = optim.Adam(vari_list, lr = 0.1)

    pi = torch.acos(torch.zeros(1)).item() * 2 

    for i in range(num_epochs):
        optimizer.zero_grad()

        V = V_c.float().cuda()
        tran = tran_c.float().cuda()
        angle = angle_c.float().cuda()
        tran_s = tran_s_cpu.float().cuda()

        angle = torch.sigmoid(angle / 10) * pi / 4
        tran_s = torch.sigmoid(tran_s) * 0.25

        R = rotation(angle, V)
        T = translation(tran, tran_s)
        
        src_temp = torch.bmm(src, R.transpose(1,2)) + T 

        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        cd_t_percent, cd_p_percent = calc_cd_percent(src_temp, src_rotated, percent = 0.3) 
        rmse_loss = cd_t + cd_p + cd_t_percent + cd_p_percent
        rmse = rmse_loss.mean()
        rmse.backward()
        optimizer.step()

    
    batch_size = batch_size // kernel
    rmse_loss = rmse_loss.reshape((batch_size, kernel))
    R = R.reshape((batch_size, kernel, 3, 3))
    T = T.reshape((batch_size, kernel, 3))
    rmse, idx = rmse_loss.topk(1, dim = 1, largest = False, sorted=True)
    
    ans = torch.zeros((batch_size, 4, 4))
    for i in range(batch_size):
        ans[i, :3, :3] = R[i, idx[i, 0]]
        ans[i,:3, 3] = T[i, idx[i, 0]] 
        ans[i,3, 3] = 1
    return ans 
# src, src_rotated, pose,  src_ori, tgt_ori, angle_t, num_epochs = 1000
# src_b, tgt_b, transform_b, src_ob, tgt_ob, angle_t
def getPredict_cd_keba_module(src, src_rotated, pose, src_ori, tgt_ori, angle_t, num_epochs = 1000):

    # batch_size = src.shape[0]
    net = cd_keba(src.shape[0])
    if torch.cuda.is_available():
        net.cuda()
        net = torch.nn.DataParallel(net)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    src = torch.unsqueeze(src, 0)
    src = src.to(device, dtype = net.module.V_c.dtype)
    for i in range(num_epochs):
        src_temp, R, T = net(src)

        src_temp = src_temp.to(dtype=torch.float32)#, R.to(dtype=torch.LongTensor), T.to(dtype=torch.LongTensor)
        cd_t, cd_p = calc_cd_percent(src_temp, src_rotated)
        rmse_loss = cd_t + cd_p
        rmse = rmse_loss.mean()

        optimizer.zero_grad()
        rmse.backward()
        optimizer.step()

        # rmse_loss = cd_t + cd_p
        # rmse = rmse_loss.mean()
        # rmse.backward()
        # optimizer.step()


    R, T = R.to(device, dtype = pose.dtype), T.to(device, dtype = pose.dtype)
    r_err = rotation_error(R, pose[:,:3,:3].transpose(1,2))
    t_err = translation_error(T.squeeze(1), pose[:,:3,3])
    print(rmse_loss)
    print(r_err)
    print(t_err)
    
    rmse, idx = rmse_loss.topk(2, largest = False, sorted=True)
    

    
    return r_err[idx[0]], t_err[idx[0]]


def translation_back(R):
    tran = 0.5 * (R - R.T)
    theta_sin = np.sqrt(tran[0][1]**2 + tran[0][2]**2 + tran[1][2]**2)
    axis = np.array([[-tran[1][2]/theta_sin, tran[0][2]/theta_sin, tran[0][1]/theta_sin]])
    axis_matrix = np.dot(axis.T, axis)
    np_matrix = R - tran - axis_matrix
    cos = np.sqrt(1. - theta_sin**2) * (np.ones((3,3)) - axis_matrix)
    result1 = sum(sum(np.square(np_matrix - cos))) # +
    result2 = sum(sum(np.square(np_matrix + cos))) # -
    if result1 >= result2:
        theta = np.pi - np.arcsin(theta_sin)
    else:
        theta = np.arcsin(theta_sin)
    return theta * 180 / np.pi, axis
