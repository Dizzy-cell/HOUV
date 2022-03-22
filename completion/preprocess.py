from IPython.terminal.interactiveshell import DISPLAY_BANNER_DEPRECATED
from numpy.core.defchararray import equal
from torch.autograd import function as F
import torch.optim as optim
import torch
# from utils.train_utils import *
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
from dataset import MVP_CP, MVP_CP_choose, Voxel
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
import glob
from torch import nn
from vis_utils import plot_grid_pcd


def dfs(idx, indx, a):
    idx = int(idx)
    if idx == -1: return
    if idx in a:
        return
    else:
        a.add(idx)
        t = indx[idx]
        indx[idx] = -1
        dfs(t, indx, a)


def bfs(idx, idx_list, vis, topk=6):
    ans = []
    pool = Queue()
    pool.put(idx)
    vis[idx] = 1
    ans.append(idx)
    while (not pool.empty()):
        t = pool.get()
        for i, a in enumerate(idx_list[t]):
            if i >= topk: break
            if a >= vis.shape[0]: continue
            if vis[a] != 0:
                continue
            pool.put(a)
            vis[a] = 1
            ans.append(a)
    return ans


def predict():
    dataset = MVP_CP(prefix="val")
    ans = dataset.input_data

    print(ans.shape)

    with h5py.File('./results.h5', 'w') as f:
        f.create_dataset('results', data=ans)
    # cur_dir = os.getcwd()
    # cmd = "zip -r submission.zip results.h5 "
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    # _, _ = process.communicate()
    # print("Submission file has been saved to submission.zip" )


def solve(class_label=0):

    dataset = MVP_CP(prefix="train")
    #dataset = MVP_CP_choose(prefix="train", idx = class_label)
    #dataset_test = MVP_CP_choose(prefix="val", idx = class_label)

    #size = dataset.__len__()
    size = 52
    indx = np.zeros(size) - 1

    sum_f1 = 0
    for i in tqdm(range(size)):
        data_a = dataset.__getitem__(i)
        a_p, a_c = data_a[1], data_a[2]

        cd_p, cd_c, f1 = calc_cd(a_p.reshape(1, 2048, 3).cuda(),
                                 a_c.reshape(1, 2048, 3).cuda(),
                                 calc_f1=True)
        sum_f1 += f1.item()
        mi = 0
        bp, bc, idx = -1, -1, -1

        q = PriorityQueue()
        for j in range(0, size):
            if i == j: continue
            data_b = dataset.__getitem__(j)
            b_p, b_c, f2 = calc_cd(a_p.unsqueeze(0).cuda(),
                                   data_b[1].unsqueeze(0).cuda(),
                                   calc_f1=True)

            q.put((f1, b_p, b_c, j))

            if f2 > mi:
                mi = f2
                bp = b_p
                bc = b_c
                idx = j

        indx[i] = idx
        from IPython import embed
        embed()


#         print("CD_P:({})/({}) F1:{}".format(cd_p.item(), cd_c.item(), f1.item()))
# print("Near Object:{}/{} CD_P:({})/({}) F1:{}".format(i, idx, bp.item(), bc.item(), mi.item()))
    print("F1:{}".format(sum_f1 / size))
    num = size // 26
    for i, j in enumerate(indx):
        if j == -1:
            continue
        a = set()
        dfs(i, indx, a)
        print(a)


def solve2(class_label=0, device='cuda:0', prefix='train'):

    #dataset = MVP_CP(prefix="train")
    dataset = MVP_CP_choose(prefix=prefix, idx=class_label)
    #dataset_test = MVP_CP_choose(prefix="val", idx = class_label)

    size = dataset.__len__()

    index_list = [[] for i in range(size)]

    sum_f1 = 0
    b_ps = torch.from_numpy(dataset.input_choose).to(device)
    for i in tqdm(range(size)):
        data_a = dataset.__getitem__(i)
        a_p, a_c = data_a[1], data_a[2]

        cd_p, cd_c, f1 = calc_cd(a_p.reshape(1, 2048, 3).cuda(),
                                 a_c.reshape(1, 2048, 3).cuda(),
                                 calc_f1=True)
        sum_f1 += f1.item()
        mi = 0
        bp, bc, idx = -1, -1, -1

        a_ps = a_p.unsqueeze(0).expand(b_ps.shape[0], -1, -1).to(device)
        cd_p, cd_c, f1 = calc_cd(a_ps, b_ps, calc_f1=True)
        cd = -1 * (cd_p + cd_c)
        values, indices = torch.topk(cd, k=6, dim=0, largest=True, sorted=True)
        index_list[i] = indices.cpu().numpy()

        print(i, indices)

    result = []
    vis = np.zeros(size)
    for i in range(size):
        if vis[i] != 0: continue
        ans = bfs(i, index_list, vis)
        result.append(ans)
        print(ans)

    # with open('preprocessData/{}_clustering.pkl'.format(class_label), 'wb') as f:
    #     pickle.dump(result, f)

    with open(
            'preprocessData/{}_knn_cd/{}_{}_{}.pkl'.format(
                prefix, class_label, 0, size), 'wb') as f:
        pickle.dump(index_list, f)

    from IPython import embed
    embed()

    return


def subsolve3(cls, ii, kk, dataset, device='cuda:0'):
    print(ii, kk, device)
    index_list = [[] for i in range(dataset.input_choose.shape[0])]
    b_ps = torch.from_numpy(dataset.input_choose).to(device)
    for i in range(ii, kk):
        data_a = dataset.__getitem__(i)
        a_p, a_c = data_a[1], data_a[2]
        a_ps = a_p.unsqueeze(0).expand(b_ps.shape[0], -1, -1).to(device)
        cd_p, cd_c, f1 = calc_cd(a_ps, b_ps, calc_f1=True)
        values, indices = torch.topk(f1, k=6, dim=0, largest=True, sorted=True)
        index_list[i] = indices.cpu().numpy()
        if i % 100 == 0:
            print("On Cuda:{} Iterater:{}/{}".format(device, i - ii, kk - ii))

    with open('preprocessData/train_knn_list/{}_{}_{}.pkl'.format(cls, ii, kk),
              'wb') as f:
        pickle.dump(index_list, f)

    return


def solve3(class_label=0, works=2):

    #dataset = MVP_CP(prefix="train")
    dataset = MVP_CP_choose(prefix="train", idx=class_label)
    #dataset_test = MVP_CP_choose(prefix="val", idx = class_label)

    size = dataset.__len__()
    works = works
    step = size // works
    assert size % works == 0

    for i in range(works):
        j = i * step
        device = "cuda:{}".format(i)
        p = mp.Process(target=subsolve3,
                       args=(class_label, j, j + step, dataset, device))
        p.start()


def combine_files(files):
    ans = []
    for filename in files:
        with open(filename, 'rb') as f:
            t = pickle.load(f)
            ans.append(t)
    result = ans[0]
    for i in range(1, len(ans)):
        result = result + ans[i]

    return result


def getSet(knn_list, topk=6):
    result = []
    size = len(knn_list)
    vis = np.zeros(size)
    for i in range(size):
        if vis[i] != 0: continue
        ans = bfs(i, knn_list, vis, topk=topk)
        result.append(ans)
    return result


def solve_knn(nclass=0,
              filename='preprocessData/train_knn_list/0_0_5200.pkl',
              prefix='train',
              device='cuda:0',
              topk=6):
    with open(filename, 'rb') as f:
        knn_list = pickle.load(f)

    dataset = MVP_CP_choose(prefix=prefix, idx=nclass)
    #cluster_set = getSet(knn_list, topk = topk)

    # from IPython import embed
    # embed()

    result = np.zeros((len(dataset), 2048, 3))
    tot = 0
    cd_tot = 0
    vox_pt_tot = 0
    for st in tqdm(knn_list):
        #if len(st) >= 8 and len(st) <= 26:
        data_list, gt_list = dataset.getData(st)

        predict, cd_mean, f1_mean, vox_pt = Dnn(data_list,
                                                gt_list,
                                                device=device)
        #predict, cd_mean, f1_mean, vox_pt, cd_eq, f1_eq= Dnn(data_list, gt_list, device = device)

        result[st[0]] = predict

        # for num in st:
        #     result[num] = predict

        tot += f1_mean
        cd_tot += cd_mean
        vox_pt_tot += vox_pt

        #print("CD:({})/({}) F1({})/({})".format(cd_mean, cd_eq, f1_mean, f1_eq))

    print("CD:{} F1:{} Vox Points:{}".format(cd_tot / len(dataset),
                                             tot / len(dataset),
                                             vox_pt_tot / len(dataset)))

    with open('preprocessData/{}_result/{}.pkl'.format(prefix, nclass),
              'wb') as f:
        pickle.dump(result, f)


def getW(vox_pts, data):
    temp = data[0]
    for i, pts in enumerate(vox_pts):
        if i > 2048: break
        temp[i] = pts
    w_ps = Variable(torch.from_numpy(temp), requires_grad=True)
    return w_ps


def Dnn(data_numpy, gt_numpy=None, device='cuda:0'):
    #logging.info(data_list.shape)
    data = torch.from_numpy(data_numpy)

    mx = np.max(data_numpy, axis=(0, 1))
    mi = np.min(data_numpy, axis=(0, 1))

    l, r = 20, 100
    while l < r:
        mid = int((l + r) // 2)
        voxel_dataset = Voxel(data_numpy.reshape(1, -1, 3),
                              grid_size=np.asarray([mid, mid, mid]),
                              max_volume_space=mx,
                              min_volume_space=mi)
        voxel_list = voxel_dataset.__getitem__(0)
        voxinds = np.unique(voxel_list[1], axis=0)
        if len(voxinds) < 2048:
            l = mid + 1
        else:
            r = mid

    # print(len(voxinds), l, r)

    vox_pts = set()
    for voxind in voxinds:
        vox_pts.add(voxel_list[0][:, voxind[0], voxind[1], voxind[2]])

    #w_ps  = getW(vox_pts, data_list)
    temp = data_numpy[0]
    #print("Voxel Points Size:{}".format(len((vox_pts))))

    for i, pts in enumerate(vox_pts):
        if i >= 2048: break
        temp[i] = pts
    w_ps = Variable(torch.from_numpy(temp), requires_grad=True)

    optimizer = optim.Adam([w_ps], lr=0.0001)

    data = data.to(device)
    w_ps = w_ps.to(device)
    ones = torch.ones((data.shape)).to(device)

    epochs = 10
    for epoch in range(epochs):
        optimizer.zero_grad()
        a = torch.mul(ones, w_ps)
        cd_p, cd_c = calc_cd(a, data, calc_f1=False)
        loss = (cd_p * 0.1 + cd_c).mean()
        loss.backward()

        optimizer.step()

    gt = torch.from_numpy(gt_numpy).to(device)
    a = torch.mul(ones, w_ps)
    cd_p, cd_c, f1 = calc_cd(a, gt, calc_f1=True)

    #cd_p_equal, cd_c_equal, f1_equal = calc_cd(data, gt, calc_f1 = True)

    #logging.info("CD:{}/{} F1:{}".format(cd_p.mean(), cd_c.mean(), f1.mean()))

    # if prefix == 0:


    return w_ps.cpu().detach().numpy(), cd_c.mean(), f1.mean(), len(
        vox_pts)  #, cd_c_equal.mean(), f1_equal.mean()


# 
def solve2_embedding(args, class_label=0, device='cuda:0', prefix='train'):

    #dataset = MVP_CP(prefix="train")
    dataset = MVP_CP_choose(prefix=prefix, idx=class_label)
    dataset_test = MVP_CP_choose(prefix="val", idx = class_label)

    #dataset_test = dataset

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args, nclasses = len(dataset) // 26)
    net.to(device)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])

    size = dataset_test.__len__()

    index_list = [[] for i in range(size)]

    sum_f1 = 0
    b_ps = torch.from_numpy(dataset_test.input_choose).to(device)

    step = 100
    epochs = b_ps.shape[0] // step
    outputes = torch.zeros(b_ps.shape[0], 200)

    F = nn.Softmax(dim=1)
    for i in range(epochs):
        _, predict = net(b_ps[i*step:i*step+step])
        outputes[i*step:i*step+step] = F(predict).detach()

    dises = torch.zeros((size))
    loss_func = nn.MSELoss()

    for i in tqdm(range(size)):
        for j in range(size):
            dises[j] = loss_func(outputes[i], outputes[j]).detach()

        values, indices = torch.topk(dises,
                                     k=6,
                                     dim=0,
                                     largest=False,
                                     sorted=True)
        index_list[i] = indices.cpu().numpy()
        
        if len(np.unique(index_list[i] // 26)) > 1:
            print(i, index_list[i])

        #     data = dataset.__getitem__(i)
        #     points_list = [data[1]]
        #     points_list2 = [data[2]]
        #     for j in indices:
        #         data = dataset.__getitem__(j)
        #         points_list.append(data[1])
        #         points_list2.append(data[2])
        #     points_list = points_list + points_list2
        #     plot_grid_pcd(points_list,
        #                   shape=[2, 7],
        #                   save_path='./Grid/grid_combine_{}.png'.format(i))
        # if i%26 ==0:
        #     t = (i // 26) * 26
        #     points_list = []
        #     points_list2 = []
        #     for j in range(26):
        #         data = dataset.__getitem__(j+t)
        #         points_list.append(data[1])
        #         points_list2.append(data[2])
                
        #     points_list = points_list 
        #     plot_grid_pcd(points_list,
        #                     shape=[2, 13],
        #                     save_path='./Grid/Ori_grid_combine_{}.png'.format(i))

    # result = []
    # vis = np.zeros(size)
    # for i in range(size):
    #     if vis[i] != 0: continue
    #     ans = bfs(i, index_list, vis)
    #     result.append(ans)
    #     print(ans)

    # with open('preprocessData/{}_clustering.pkl'.format(class_label), 'wb') as f:
    #     pickle.dump(result, f)

    with open(
            'preprocessData/{}_knn_cd/{}_{}_{}.pkl'.format(
                prefix, class_label, 0, size), 'wb') as f:
        pickle.dump(index_list, f)

    # from IPython import embed
    # embed()

    return


def getDis(a, b):
    dis = torch.zeros(a.shape[0])

    for i in range(a.shape[0]):
        dis[i] = (a[i] - b[i]) * (a[i] - b[i])
    res = dis.sum()
    return res


def solve2_describe(args, class_label=0, device='cuda:0', prefix='train'):

    #dataset = MVP_CP(prefix="train")
    dataset = MVP_CP_choose(prefix=prefix, idx=class_label)
    #dataset_test = MVP_CP_choose(prefix="val", idx = class_label)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = model_module.Model(args)
    net.to(device)

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])

    size = dataset.__len__()

    index_list = [[] for i in range(size)]

    sum_f1 = 0
    b_ps = torch.from_numpy(dataset.input_choose).to(device)

    b_mx = torch.max(b_ps, dim=1)[0]
    b_mi = torch.min(b_ps, dim=1)[0]

    b_des = torch.cat([b_mx, b_mi], dim=1)

    dises = torch.zeros((size))

    for i in tqdm(range(size)):

        dises = (b_des - b_des[i]) * (b_des - b_des[i])
        dises, _ = torch.topk(dises, k = 6, dim = 1, largest = False, sorted = True)

        dises = dises.sum(dim=1)

        values, indices = torch.topk(dises,
                                     k=6,
                                     dim=0,
                                     largest=False,
                                     sorted=True)
        index_list[i] = indices.cpu().numpy()

        indices = index_list[i] // 26


        # visulation
        if len(np.unique(indices)) > 1:
            print(i, np.unique(indices))
            data = dataset.__getitem__(i)
            points_list = [data[1]]
            points_list2 = [data[2]]
            for j in indices:
                data = dataset.__getitem__(j)
                points_list.append(data[1])
                points_list2.append(data[2])
            points_list = points_list + points_list2
            plot_grid_pcd(points_list, shape= [2,7], save_path='./Grid/grid_combine_{}.png'.format(i))

    with open(
            'preprocessData/{}_knn_dis/{}_{}_{}.pkl'.format(
                prefix, class_label, 0, size), 'wb') as f:
        pickle.dump(index_list, f)

    # from IPython import embed
    # embed()

    return


def produceResult(prefix='train', device = 'cuda:0'):
    numpy_lst = []
    for i in range(0, 16):
        with open('preprocessData/{}_result/{}.pkl'.format(prefix, i),
                  'rb') as f:
            data = pickle.load(f)
            numpy_lst.append(data)
    dataset = MVP_CP(prefix=prefix)

    result = getResult(dataset, numpy_lst)
    print("Result Shape:", result.shape)

    step = 1000
    epochs = len(dataset) // step
    cd_p_sum = 0
    f1_sum = 0
    cd_p_sum_ori = 0
    f1_sum_ori = 0
    for epoch in range(epochs):
        index = np.array(list(range(epoch * step, epoch * step + step)))
        index2 = index // 26

        pri = torch.from_numpy(result[index]).type(torch.float32).to(device)
        ori = torch.from_numpy(dataset.input_data[index]).to(device)
        gt = torch.from_numpy(dataset.gt_data[index2]).to(device)

        cd_c, cd_p, f1 = calc_cd(pri, gt, calc_f1=True)
        cd_c_ori, cd_p_ori, f1_ori = calc_cd(ori, gt, calc_f1=True)

        cd_p_sum += (cd_p.sum()).item()
        f1_sum += (f1.sum()).item()

        cd_p_sum_ori += (cd_p_ori.sum()).item()
        f1_sum_ori += (f1_ori.sum()).item()

    print("CD:{} F1:{}".format(cd_p_sum / len(dataset), f1_sum / len(dataset)))
    print("Origin CD:{} F1:{}".format(cd_p_sum_ori / len(dataset),
                                      f1_sum_ori / len(dataset)))

    with h5py.File('preprocessData/{}_result/results.h5'.format(prefix),
                   'w') as f:
        f.create_dataset('results', data=result)
    
    return 


def combineDataset(nclass = 0, device = 'cuda:0'):
    print("Solve:", nclass)
    dataset_train = MVP_CP_choose(prefix = 'train', idx = nclass)
    dataset_val = MVP_CP_choose(prefix = 'val', idx = nclass)

    aa, b = dataset_train.gt_data, dataset_val.gt_data
    
    a = np.concatenate([aa,b], axis = 0)
    c = torch.from_numpy(a).to(device)
    knns = np.zeros((c.shape[0], 6), dtype=np.int)

    cd_c_tot = 0
    for i in tqdm(range(aa.shape[0], c.shape[0])):
        t = c[i]
        b = t.reshape((1,2048,3)).expand(aa.shape[0], -1, -1)
        cd_p, cd_c = calc_cd(c[:aa.shape[0]], b, calc_f1 = False)
        values, indices = torch.topk(cd_c + cd_p, k =6, dim=0, largest = False, sorted = True)

        knns[i] = indices.cpu().numpy()

        pts = []
        for j in range(6):
            pts.append(a[knns[i][j]])
        
        cd_c_tot  += cd_c[indices[0]].item()

        # plot_grid_pcd(pts, shape=(1,6), save_path='./Grid/Train_test_GT_{}.png'.format(i))
        # print(cd_c[indices[1]], knns[i])



    print(cd_c_tot / (c.shape[0] - aa.shape[0]))


    


    return 

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

    # for i in range(1, 16):
    #     combineDataset(nclass = i)

    # solve2(1, device= 'cuda:1')
    # solve3(0, works = 4)
    # solve3(1, works = 4)

    solve2_embedding(args, 1, device = 'cuda:0')
    #solve2_describe(args, 1, device = 'cuda:0')

    # l, r = 1, 4
    # #l, r = 4,8
    # #l, r = 8,12
    # # #l, r = 12,16
    # l, r = 0, 16
    # for i in range(l, r):
    #     solve2_describe(args, i, prefix='train', device='cuda:0')

    # prefix = 'val'
    # for i in range(0,16):
    #     filename = glob.glob("preprocessData/{}_knn_dis/{}_*.pkl".format(prefix, i))[0]
    #     print("Solve {}:{}".format(prefix, i))
    #     solve_knn(i, filename = filename , prefix = prefix , topk = 5, device = 'cuda:2') # topk not used!

    # produce the result
    # numpy_lst = []
    # for i in range(0, 16):
    #     with open('preprocessData/train_result/{}.pkl'.format(i), 'rb') as f:
    #         data = pickle.load(f)
    #         numpy_lst.append(data)
    # dataset = MVP_CP(prefix='train')

    # result = getResult(dataset, numpy_lst)
    # print("Result Shape:", result.shape)

    # step = 1000
    # epochs = len(dataset) // step
    # cd_p_sum = 0
    # f1_sum = 0
    # cd_p_sum_ori = 0
    # f1_sum_ori = 0
    # for epoch in range(epochs):
    #     index = np.array(list(range(epoch * step, epoch * step + step)))
    #     index2 = index // 26

    #     pri = torch.from_numpy(result[index]).type(torch.float32).to('cuda:0')
    #     ori = torch.from_numpy(dataset.input_data[index]).to('cuda:0')
    #     gt = torch.from_numpy(dataset.gt_data[index2]).to('cuda:0')

    #     cd_c, cd_p, f1 = calc_cd(pri, gt, calc_f1=True)
    #     cd_c_ori, cd_p_ori, f1_ori = calc_cd(ori, gt, calc_f1=True)

    #     cd_p_sum += (cd_p.sum()).item()
    #     f1_sum += (f1.sum()).item()

    #     cd_p_sum_ori += (cd_p_ori.sum()).item()
    #     f1_sum_ori += (f1_ori.sum()).item()

    # print("CD:{} F1:{}".format(cd_p_sum / len(dataset), f1_sum / len(dataset)))
    # print("Origin CD:{} F1:{}".format(cd_p_sum_ori / len(dataset), f1_sum_ori / len(dataset)))





    # produceResult(prefix = 'val', device = 'cuda:0')

    # from IPython import embed
    # embed()

    # with h5py.File('preprocessData/train_result/results.h5','w') as f:
    #     f.create_dataset('results', data = result)

    # for i in range(1, 16):
    #     solve2(i, prefix = 'val', device = 'cuda:3')
    # filename = glob.glob("preprocessData/val_knn_list/{}_*.pkl".format(i))[0]
    # solve_knn(i, filename = filename , prefix = 'val', device= 'cuda:3')

    # for i in range(16):
    #     solve2(i, prefix = 'test')
    #     filename = glob.glob("preprocessData/test_knn_list/{}_*.pkl".format(i))[0]
    #     solve_knn(i, filename = filename , prefix = 'test')

    #predict()
