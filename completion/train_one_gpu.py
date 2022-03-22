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
from dataset import MVP_CP, MVP_CP_voxel_point, MVP_CP_choose
from models.vox_asym import build_model
from utils_loss import getGt, getPts_2048, getPts_50
from model_utils import gen_grid_up, calc_emd, calc_cd
from vis_utils import plot_single_pcd, plot_combine_pcd
import spconv
from tqdm import tqdm
import importlib
import warnings

warnings.filterwarnings("ignore")

def train():
    device = torch.device('cuda:0')
    logging.info(str(args))
    if args.eval_emd:
        metrics = ['cd_p', 'cd_t', 'emd', 'f1']
    else:
        metrics = ['cd_p', 'cd_t', 'f1']
    
    best_epoch_losses = {
        m: (0, 0) if m == 'f1' else (0, math.inf)
        for m in metrics
    }
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataset = MVP_CP_choose(idx = 1, prefix="train")
    dataset_test = MVP_CP_choose(idx = 1, prefix="val")

    # from IPython import embed
    # print("In train_one_gpu.py!")

    # for i in range(0,260,26):
    #     data = dataset_test.__getitem__(i)
    #     points = data[-1]
    #     points_vis = data[-2]
    #     # plot_single_pcd(points, 'visuilation/{}.png'.format(i))
    #     # plot_single_pcd(points_vis, 'visuilation/{}_vis.png'.format(i))
    #     plot_combine_pcd(points, points_vis, 'visuilation/ID{}_{}_combine.png'.format(1, i))
    
    #     #Net_vis(points, points_vis)

    # embed()

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=int(args.workers))
    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=int(
                                                      args.workers))
    logging.info('Length of train dataset:%d', len(dataset))
    logging.info('Length of test dataset:%d', len(dataset_test))

    net = build_model(args)
    net.to(device)

    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError(
                'lr_decay_interval and lr_step_decay_epochs are mutually exclusive!'
            )
        if args.lr_step_decay_epochs:
            decay_epoch_list = [
                int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')
            ]
            decay_rate_list = [
                float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')
            ]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.parameters(),
                              lr=lr,
                              initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.parameters(),
                              lr=lr,
                              weight_decay=args.weight_decay,
                              betas=betas)

    alpha = None
    if args.varying_constant:
        varying_constant_epochs = [
            int(ep.strip()) for ep in args.varying_constant_epochs.split(',')
        ]
        varying_constant = [
            float(c.strip()) for c in args.varying_constant.split(',')
        ]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    if args.load_model:
        ckpt = torch.load(args.load_model)
        net.load_state_dict(ckpt['net_state_dict'])
        logging.info("%s's previous weights loaded." % args.model_name)

    for epoch in range(args.start_epoch, args.nepoch):
        train_loss_meter.reset()
        net.train()

        if args.varying_constant:
            for ind, ep in enumerate(varying_constant_epochs):
                if epoch < ep:
                    alpha = varying_constant[ind]
                    break
                elif ind == len(varying_constant_epochs) - 1 and epoch >= ep:
                    alpha = varying_constant[ind + 1]
                    break

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_func = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()
            _, inputs, gt = data
            
            vox_pos, vox_idx, vox_fea = inputs  
            vox_pos = vox_pos.type(torch.FloatTensor).to(device)
            vox_idx = vox_idx.type(torch.LongTensor).to(device)
            vox_fea = vox_fea.type(torch.FloatTensor).to(device)

            out_voxel, out_bias_pos = net(vox_fea, vox_idx)  # out_voxel == out_predict
            out_pts_2048 = getPts_2048(out_voxel, vox_pos, out_bias_pos)
            #out_pts_50 = getPts_50(out_voxel, out_vox_pos)

            gt_pos, gt_idx, gt_fea = gt  
            gt_pos = gt_pos.type(torch.FloatTensor).to(device)
            gt_idx = gt_idx.type(torch.LongTensor).to(device)
            gt_fea = gt_fea.type(torch.FloatTensor).to(device)

            gt_vox_one_hot, gt_vox_pos, gt_pts = getGt(gt_fea, gt_idx)

            loss_predict = loss_func(out_voxel, gt_vox_one_hot) 
            loss_predict.backward()

            optimizer.step()

            # metric_emb = calc_emd(out_pts_2048, gt_pts) # Loss no grad_fn
            # metric_cd = calc_cd(out_pts_2048, gt_pts)

            train_loss_meter.update(loss_predict.item())

            if i % args.step_interval_to_print == 0:
                logging.info(
                    exp_name +
                    ' train [%d: %d/%d]  loss_type: %s, loss_predict: %.3f lr: %f'
                    % (epoch, i, len(dataset) / args.batch_size, args.loss,
                       loss_predict.mean().item(), lr) +
                    ' alpha: ' + str(alpha))

        if epoch % args.epoch_interval_to_save == 0:
            save_model('%s/network.pth' % log_dir, net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            val(net, epoch, val_loss_meters, dataloader_test,
                best_epoch_losses)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test,
        best_epoch_losses, device = 'cuda:0'):
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.eval()

    loss_func = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            _, inputs, gt = data

            vox_pos, vox_idx, vox_fea = inputs  
            vox_pos = vox_pos.type(torch.FloatTensor).to(device)
            vox_idx = vox_idx.type(torch.LongTensor).to(device)
            vox_fea = vox_fea.type(torch.FloatTensor).to(device)

            out_voxel, out_bias_pos = net(vox_fea, vox_idx)  # out_voxel == out_predict
            out_pts_2048 = getPts_2048(out_voxel, vox_pos, out_bias_pos)
            #out_pts_50 = getPts_50(out_voxel, out_vox_pos)

            gt_pos, gt_idx, gt_fea = gt  
            gt_pos = gt_pos.type(torch.FloatTensor).to(device)
            gt_idx = gt_idx.type(torch.LongTensor).to(device)
            gt_fea = gt_fea.type(torch.FloatTensor).to(device)

            gt_vox_one_hot, gt_vox_pos, gt_pts = getGt(gt_fea, gt_idx)

            # metric_emb = calc_emd(out_pts_2048, gt_pts) # Loss no grad_fn

            loss_predict = loss_func(out_voxel, gt_vox_one_hot) 
            metircs = calc_cd(out_pts_2048, gt_pts, calc_f1=True) 
            #cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)

            for k, v in val_loss_meters.items():
                v.update(metircs[k].item(), out_voxel.shape[0])

        fmt = 'best_%s: %f [epoch %d]; '
        best_log = ''
        for loss_type, (curr_best_epoch,
                        curr_best_loss) in best_epoch_losses.items():
            if (val_loss_meters[loss_type].avg < curr_best_loss and loss_type != 'f1') or \
                    (val_loss_meters[loss_type].avg > curr_best_loss and loss_type == 'f1'):
                best_epoch_losses[loss_type] = (curr_epoch_num,
                                                val_loss_meters[loss_type].avg)
                save_model('%s/best_%s_network.pth' % (log_dir, loss_type),
                           net)
                logging.info('Best %s net saved!' % loss_type)
                best_log += fmt % (loss_type, best_epoch_losses[loss_type][1],
                                   best_epoch_losses[loss_type][0])
            else:
                best_log += fmt % (loss_type, curr_best_loss, curr_best_epoch)

        curr_log = ''
        for loss_type, meter in val_loss_meters.items():
            curr_log += 'curr_%s: %f; ' % (loss_type, meter.avg)

        logging.info(curr_log)
        logging.info(best_log)


if __name__ == "__main__":
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
    
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    train()
