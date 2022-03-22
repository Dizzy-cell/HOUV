import argparse
#from typing_extensions import Required
import numpy as np
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
from time import time
# from torch.utils.data import DataLoader, Subset

import torch.optim as optim
from tqdm import tqdm
import os
import random
import sys

import logging
import math
import importlib
import datetime
import munch
import yaml
import argparse
import copy

from train_utils import AverageValueMeter, save_model, solve
from dataset import MVP_RG, MVP_RG_rotated
from visu_utils import plot_grid_pcd

import torch
from torch.autograd import Variable
import  torch.optim as optim
from torch.autograd import function as F
import numpy as np

from train_utils import getPredict, getPredict_cd, getPredict_cd_keba, getPredict_cd_keba_module, getPredict_cd_keba_v2, getPredict_cd_keba_v3, getPredict_cd_vox, solve  #getPredict_cd_keba_part_v3

from train_utils import rotation_error, translation_error, rmse_loss

import open3d as o3d
from visu_utils import analyseDis, getDis, analyseDises

#import time

# def Dnn(data, device = 'cuda:0'):   

#     A, B = data[0], data[1]
#     w_ps = Variable(torch.zeros(A.shape), requires_grad = True))
#     optimizer = optim.Adam([w_ps], lr = 0.001)

#     A = A.to(device)
#     B = B.to(device)
#     ones = torch.ones((A.shape)).to(device)

#     epochs = 100
#     for epoch in range(epochs):
#         optimizer.zero_grad()


def train():
    logging.info(str(args))
    metrics = ['RotE', 'transE', 'MSE', 'RMSE', 'recall']
    best_epoch_losses = {m: (0, 0) if m =='recall' else (0, math.inf) for m in metrics}
    # best_epoch_losses = (0, inf)
    train_loss_meter = {m: AverageValueMeter() for m in metrics}
    val_loss_meters = {m: AverageValueMeter() for m in metrics}
    val_split_loss_meters = []
    for i in range(args.num_rot_levels):
        row = []
        for j in range(args.num_corr_levels):
            row.append(copy.deepcopy(val_loss_meters))
        val_split_loss_meters.append(row)
    val_split_loss_meters = np.array(val_split_loss_meters)

    dataset = MVP_RG_rotated(prefix="train", args=args)
    dataset_test = MVP_RG_rotated(prefix="val", args=args)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    
    dataloader = dataloader_test

    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    andis = []

    for epoch in range(args.start_epoch, args.nepoch):
        for v in train_loss_meter.values():
            v.reset()

        for i, data in enumerate(dataloader, 0):

            src, tgt, src_rotated,  tgt_rotated, transform, match_level, rot_level, pose1, pose2, angle_t, label, src_vox, tgt_vox, src_vox_len, tgt_vox_len, add_ps, angle = data

            src = src.float().cuda()
            tgt = tgt.float().cuda()
            src_rotated = src_rotated.float().cuda()
            tgt_rotated = tgt_rotated.float().cuda()
            transform = transform.float().cuda()
            pose1 = pose1.float().cuda()
            angle_t = angle_t.float().cuda()
            label = label.long().cuda()
            
            import time
            start = time.time()
            #r_err, t_err, ans = solve(src_rotated, tgt_rotated, transform, src, tgt, label, add_ps)
            
            threshold = 0.02
            trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
            
            # open3d 0.9.0
#             from IPython import embed
#             embed()
            
            batch_size = src.shape[0]
            ans = np.zeros((batch_size, 4, 4))
            
            #ICP
            # for ii in range(batch_size):
            #     trans_init = np.asarray([[0.862, 0.011, -0.507, 0],
            #              [-0.139, 0.967, -0.215, 0],
            #              [0.487, 0.255, 0.835, 0], [0.0, 0.0, 0.0, 1.0]])
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(src_rotated[ii].cpu().numpy())
                
            #     pcd2 = o3d.geometry.PointCloud()
            #     pcd2.points = o3d.utility.Vector3dVector(tgt_rotated[ii].cpu().numpy())
                
                
            #     reg_p2p = o3d.registration.registration_icp(
            #         pcd, pcd2, threshold, trans_init,
            #         o3d.registration.TransformationEstimationPointToPoint(),
            #         o3d.registration.ICPConvergenceCriteria(max_iteration = 500))
                
            #     ans[ii] = reg_p2p.transformation
            
            #GO-icp
            from py_goicp import GoICP, POINT3D;
            goicp = GoICP()
            for ii in range(batch_size):
                print(ii)
                pcd = o3d.geometry.PointCloud()
                pcd = src_rotated[ii].cpu().numpy()
                pcd /= 2.5
                pcd_list = []
                for j in range(pcd.shape[0]):
                    pcd_list.append(POINT3D(float(pcd[j][0]),float(pcd[j][1]),float(pcd[j][2])))

                pcd2 = o3d.geometry.PointCloud()
                pcd2 = tgt_rotated[ii].cpu().numpy()
                pcd2 /= 2.5
                pcd2_list = []
                for j in range(pcd2.shape[0]):
                    pcd2_list.append(POINT3D(float(pcd2[j][0]),float(pcd2[j][1]),float(pcd2[j][2])))

                goicp.loadModelAndData(2048,pcd_list, 2048, pcd2_list)
                goicp.setDTSizeAndFactor(300,2.0)
                goicp.BuildDT()
                goicp.Register()

                ans[ii][:3,:3] = np.array(goicp.optimalRotation())
                ans[ii][:3,3] = np.array(goicp.optimalTranslation()) * 2.5
                # from IPython import embed
                # embed()

            # FGR
#             radius_feature = 0.05
#             radius_normal = 0.025
#             distance_threshold = 0.1
            
#             for ii in range(batch_size):
#                 trans_init = np.asarray([[0.862, 0.011, -0.507, 0],
#                          [-0.139, 0.967, -0.215, 0],
#                          [0.487, 0.255, 0.835, 0], [0.0, 0.0, 0.0, 1.0]])
                
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(src_rotated[ii].cpu().numpy())
#                 pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))
#                 pcd_fpfh = o3d.registration.compute_fpfh_feature(
#                     pcd,
#                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=60))
                
#                 pcd2 = o3d.geometry.PointCloud()
#                 pcd2.points = o3d.utility.Vector3dVector(tgt_rotated[ii].cpu().numpy())
#                 pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=20))
                
#                 pcd2_fpfh = o3d.registration.compute_fpfh_feature(
#                     pcd2,
#                     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=60))
                
#                 reg_p2p = o3d.registration.registration_fast_based_on_feature_matching(
#                         pcd, pcd2, pcd_fpfh, pcd2_fpfh,
#                         o3d.registration.FastGlobalRegistrationOption(
#                             maximum_correspondence_distance=distance_threshold, iteration_number = 500))
                
#                 ans[ii] = reg_p2p.transformation
            
            ans = torch.from_numpy(ans).float().cuda()
            r_err = rotation_error(ans[:, :3, :3], transform[:, :3, :3])
            t_err = translation_error(ans[:,:3,3], transform[:,:3,3])
            mse = rmse_loss(src_rotated, ans, transform)

            end = time.time()
            print(end - start)

            train_loss_meter['RotE'].update(r_err.mean().item())
            train_loss_meter['transE'].update(t_err.mean().item())
            train_loss_meter['MSE'].update(mse.mean().item())
            
            andis.append(ans[:,:3,3].cpu().numpy())

            # translation_back(ans[j,:3,:3].detach().cpu().numpy())
            # flags = torch.nonzero((add_ps.reshape(-1) == 1))
            # print("ADD process:", flags)
            # print(angle[flags])

            if i % 10 == 0:
                logging.info('RotE:{} TransE:{} MSE:{}'.format(train_loss_meter['RotE'].avg, train_loss_meter['transE'].avg, train_loss_meter['MSE'].avg))

            # from IPython import embed
            # embed()

        if epoch % args.epoch_interval_to_save == 0:
            #save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

        print(train_loss_meter['RotE'].avg)
        print(train_loss_meter['transE'].avg)
        print(train_loss_meter['MSE'].avg)


        dis = np.vstack(andis)
        analyseDis(dis)


        dises = [dis]

        dis_sigmoid = getDis('results/results_sigmoid.h5')
        dises.append(dis_sigmoid)

        dis_sine = getDis('results/results_sine.h5')
        dises.append(dis_sine)
    
        analyseDises(dises)

        break

        # if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
        #     val(net, epoch, val_loss_meters, val_split_loss_meters, dataloader_test, best_epoch_losses)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.benchmark + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    train()


