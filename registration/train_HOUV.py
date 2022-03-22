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
from dataset import MVP_RG, MVP_RG_rotated, ModelNet
from visu_utils import plot_grid_pcd

import torch
from torch.autograd import Variable
import  torch.optim as optim
from torch.autograd import function as F
import numpy as np

from train_utils import getPredict, getPredict_cd, getPredict_cd_keba, getPredict_cd_keba_module, getPredict_cd_keba_v2, getPredict_cd_keba_v3, getPredict_cd_vox, solve  #getPredict_cd_keba_part_v3

from train_utils import rotation_error, translation_error, rmse_loss
from models.houv import HOUV, predict_model, solve_model

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

    # dataset = ModelNet(name = 'clean')               # other dataset
    # dataset_test = ModelNet(name = 'noisy')

    # dataset = ModelNet(name = 'icl_nuim')
    # dataset_test = ModelNet(name = 'icl_nuim')
    
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

    net = HOUV(args.batch_size * args.kernel, 0)
    net.cuda()

    for v in train_loss_meter.values():
        v.reset()

    for i, data in enumerate(dataloader, 0):

        src, tgt, src_rotated,  tgt_rotated, transform, match_level, rot_level, pose1, pose2, angle_t, label, src_vox, tgt_vox, src_vox_len, tgt_vox_len, add_ps, angle = data

        #src_rotated, tgt_rotated, transform = data       # other dataset
        # src, tgt = src_rotated, tgt_rotated
        # label = None
        # add_ps = None


        src = src.float().cuda()
        tgt = tgt.float().cuda()
        src_rotated = src_rotated.float().cuda()
        tgt_rotated = tgt_rotated.float().cuda()
        transform = transform.float().cuda()

        #import time
        #start = time.time()
        #r_err, t_err, ans = solve(src_rotated, tgt_rotated, transform, src, tgt, label, add_ps, kernel = 32)
        #end = time.time()
        #print(end - start)

        r_err, t_err, ans = solve_model(net, src_rotated, tgt_rotated, transform, src, tgt, angle, add_ps)

        mse = rmse_loss(src_rotated, ans, transform)

        train_loss_meter['RotE'].update(r_err.mean().item())
        train_loss_meter['transE'].update(t_err.mean().item())
        train_loss_meter['MSE'].update(mse.mean().item())

        # translation_back(ans[j,:3,:3].detach().cpu().numpy())
        # flags = torch.nonzero((add_ps.reshape(-1) == 1))
        # print("ADD process:", flags)
        # print(angle[flags])

        if i % 10 == 0:
            logging.info('RotE:{} TransE:{} MSE:{}'.format(train_loss_meter['RotE'].avg, train_loss_meter['transE'].avg, train_loss_meter['MSE'].avg))

    print(train_loss_meter['RotE'].avg)
    print(train_loss_meter['transE'].avg)
    print(train_loss_meter['MSE'].avg)




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


