import argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
import subprocess

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

from train_utils import AverageValueMeter, save_model, getPredict_test_keba, getPredict_test_keba_v2, solve
from dataset import MVP_RG, MVP_RG_rotated, MVP_RG_rotated_bound, Modelnet_RG_rotated_bound
from tqdm import tqdm

def test():
    logging.info(str(args))

    dataset_test = Modelnet_RG_rotated_bound(prefix="clean", args=args)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                            shuffle=False, num_workers=int(args.workers))
    logging.info('Length of test dataset:%d', len(dataset_test))

    print("Test Size:{}".format(len(dataloader_test)))

    result_list = []
    for i, data in enumerate(dataloader_test):
        src, tgt, label = data
        src = src.float().cuda()
        tgt = tgt.float().cuda()

        result = solve(src, tgt, prefix = 'test', )
        result_list.append(result.detach().numpy())

        print('Solve step:{}'.format(i))

    all_results = np.concatenate(result_list, axis=0)
    print(all_results.shape, args.l, args.r)

    filename = log_dir + '/{}_{}.npy'.format(args.l, args.r)
    np.save(filename, all_results)

    return

def combine(step = 500, num = 4):
    ans = []
    for i in range(num):
        filename = log_dir + '/{}_{}.npy'.format(step * i, step * i + step)
        ans.append(np.load(filename))
    all_results = np.concatenate(ans, axis = 0)
    print(all_results.shape)

    with h5py.File(log_dir + '/results.h5', 'w') as f:
        f.create_dataset('results', data=all_results)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    
    parser.add_argument('-l', '--left', default=0, help ='solve the left label')
    parser.add_argument('-r', '--right', default=16, help = 'sovle the right label')
    parser.add_argument( '--combine', default=False, help = 'combine the results')

    arg = parser.parse_args()
    config_path = arg.config
    
    args = munch.munchify(yaml.safe_load(open(config_path)))

    args.l = int(arg.left)
    args.r = int(arg.right)
    args.combine = arg.combine

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    if args.combine:
        combine(step = 500, num = 4)
    else:
        test()


