
from torch.autograd import function as F
import torch.optim as optim
import torch
from train_utils import *

import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import MVP_CP, MVP_CP_choose, Voxel, MVP_CP_choose_triple
from model_utils import gen_grid_up, calc_emd, calc_cd
import h5py
import subprocess
from torch.autograd import Variable

import warnings

warnings.filterwarnings("ignore")

from tqdm import tqdm
import numpy as np

from queue import PriorityQueue, Queue 
import pickle
import multiprocessing as mp
# import glob

# from .models import model_embedding
from torch import nn


def embedding(args, nclass = 1, prefix = 'train', log_dir = 'log_dir', device = 'cuda:0'):
    dataset = MVP_CP_choose_triple(prefix= 'train', idx = nclass)
    dataset_test = MVP_CP_choose_triple(prefix= 'val', idx = nclass)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 16, shuffle = True, num_workers = args.workers)
    dataloader_test =  torch.utils.data.DataLoader(dataset_test, batch_size = 16, shuffle = False, num_workers = args.workers)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args, nclasses = len(dataset) // 26)
    net.to(device)
    print(len(dataset)//26)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])

    optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    loss_func = nn.MSELoss() 
    loss_entropy = nn.CrossEntropyLoss()
    
    best_loss = 1
    for epoch in range(args.nepoch):
        loss1Avg = AverageValueMeter()
        loss2Avg = AverageValueMeter()
        lossAvg = AverageValueMeter()
        lossEnt = AverageValueMeter()

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            a_ori, a_nerb , a_other, a_com, labels = data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device),  data[5].to(device)

            emb, pre = net(a_ori)
            # emb_nerb, _ = net(a_nerb)
            # emb_other, _ = net(a_other)

            # loss1 = loss_func(emb, emb_nerb)
            # loss2 = loss_func(emb, emb_other) + loss_func(emb_nerb, emb_other)
            # loss = loss1 - loss2 + 0.1

            loss_pre = loss_entropy(pre, labels)
            loss = loss_pre

            loss.backward()

            optimizer.step()

            loss1Avg.update(loss.item())
            loss2Avg.update(loss.item())
            lossAvg.update(loss.item())
            lossEnt.update(loss_pre.item())

        print(epoch, loss1Avg.avg, loss2Avg.avg, lossAvg.avg, lossEnt.avg)

        if epoch % args.epoch_interval_to_save == 0:
            save_model_nonmodule('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")
        
        # if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
        #     loss = val(net, dataloader_test, device = device)

        #     if loss < best_loss:
        #         best_loss = loss
        #         save_model_nonmodule('%s/best_loss_network.pth' % log_dir, net)

        if lossEnt.avg < best_loss:
            best_loss = lossEnt.avg
            save_model_nonmodule('%s/best_loss_network.pth' % log_dir, net)
            logging.info("Saving best loss net...")


def val(net, dataloader_test, device = 'cuda:0'):
    net.eval()
    loss_func = nn.MSELoss()
    loss_entropy = nn.CrossEntropyLoss()

    loss1Avg = AverageValueMeter()
    loss2Avg = AverageValueMeter()
    lossAvg = AverageValueMeter()
    lossEnt = AverageValueMeter()


    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            a_ori, a_nerb , a_other, a_com, labels = data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device), data[5].to(device)
            
            emb, pre = net(a_ori)

            lossent = loss_entropy(pre, labels)

            lossEnt.update(lossent.item())

        print("Validation:", loss1Avg.avg, loss2Avg.avg, lossAvg.avg, lossEnt.avg)

    net.train()
    return lossEnt.avg




if __name__ == "__main__":
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c',
                        '--config',
                        help='path to config file',
                        required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    time = datetime.datetime.now().isoformat()[:19]
    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        log_dir = os.path.dirname(args.load_model)
    else:
        exp_name = args.model_name + '_' + args.loss + '_' + args.flag + '_' + time
        log_dir = os.path.join(args.work_dir, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(
                                os.path.join(log_dir, 'train.log')),
                            logging.StreamHandler(sys.stdout)
                        ])

    #solve2(1, device= 'cuda:1')
    # solve3(0, works = 4)
    # solve3(1, works = 4)

    embedding(args, nclass = 1, log_dir = log_dir, device = 'cuda:1')

    # l, r = 1, 4
    # #l, r = 4,8
    # #l, r = 8,12
    # # #l, r = 12,16
    # l, r = 2,16
    # for i in range(l, r):
    #     solve2(i, prefix = 'train', device = 'cuda:{}'.format(l // 4))

    # for i in range(0,16):
    #     filename = glob.glob("preprocessData/train_knn_cd/{}_*.pkl".format(i))[0]
    #     solve_knn(i, filename = filename , prefix = 'train', topk = 5, device = 'cuda:1')
    

    # for i in range(1, 16):
    #     solve2(i, prefix = 'val', device = 'cuda:3')
        # filename = glob.glob("preprocessData/val_knn_list/{}_*.pkl".format(i))[0]
        # solve_knn(i, filename = filename , prefix = 'val', device= 'cuda:3')
    

    # for i in range(16):
    #     solve2(i, prefix = 'test')
    #     filename = glob.glob("preprocessData/test_knn_list/{}_*.pkl".format(i))[0]
    #     solve_knn(i, filename = filename , prefix = 'test')

    #predict()
