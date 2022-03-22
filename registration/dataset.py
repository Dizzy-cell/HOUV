import enum
import random
import h5py
import numpy as np
import os
import open3d as o3d
import torch
from torch.utils.data import Dataset
from train_utils import translation_back

def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R, angle = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0), angle


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R, angle


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)

def random_pose_lr(max_angle, min_angle, max_trans):
    R, angle = random_rotation_lr(max_angle, min_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0), angle


def random_rotation_lr(max_angle, min_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = min_angle + np.random.rand() * (max_angle - min_angle)
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R, angle

class MVP_RG(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('./data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][:].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][:]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))

                self.pose_src = np.array(f['pose_src'][:].astype('float32'))
                self.pose_tgt = np.array(f['pose_tgt'][:].astype('float32'))
                self.complete =  np.array(f['complete'][:].astype('float32'))

                self.ori_src = np.array(f['src'][:].astype('float32'))
                self.ori_tgt = np.array(f['tgt'][:].astype('float32'))
        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]

            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]

        if self.prefix == "train":
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]

        # src = np.random.permutation(src)
        # tgt = np.random.permutation(tgt)

        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)

        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            match_level = match_level
            rot_level = rot_level
            return src, tgt, transform, match_level, rot_level
        else:
            return src, tgt


class ModelNet(Dataset):
    "docstring for ModelNet"
    def __init__(self, name = 'clean'):
        self.name = name
        if self.name == 'clean':
            with h5py.File('./data/modelnet_clean.h5', 'r') as f:
                self.source = f['source'][...]
                self.target = f['target'][...]
                self.transform = f['transform'][...]
        elif self.name == 'noisy':
            with h5py.File('./data/modelnet_noisy.h5', 'r') as f:
                self.source = f['source'][...]
                self.target = f['target'][...]
                self.transform = f['transform'][...]
        elif self.name == 'unseen':
            with h5py.File('./data/modelnet_unseen.h5', 'r') as f:
                self.source = f['source'][...]
                self.target = f['target'][...]
                self.transform = f['transform'][...]
        elif self.name == 'icl_nuim':
            with h5py.File('./data/icl_nuim.h5', 'r') as f:
                self.source = f['source'][...]
                self.target = f['target'][...]
                self.transform = f['transform'][...]     
        self.n_points = 1024

    def __getitem__(self, index):
        pcd1 = self.source[index][:self.n_points]
        pcd2 = self.target[index][:self.n_points]
        transform = self.transform[index]
        return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32')

    def __len__(self):
        return self.transform.shape[0]

class MVP_RG_rotated(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('./data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][:].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][:]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                self.src_rotated = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt_rotated = np.array(f['rotated_tgt'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))

                self.pose_src = np.array(f['pose_src'][:].astype('float32'))
                self.pose_tgt = np.array(f['pose_tgt'][:].astype('float32'))
                self.complete =  np.array(f['complete'][:].astype('float32'))

                self.ori_src = np.array(f['src'][:].astype('float32'))
                self.ori_tgt = np.array(f['tgt'][:].astype('float32'))
        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]

            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)
        # from IPython import embed
        # print(self.prefix)
        # #translation_back(self.transforms[j,:3,:3])
        # embed()

        self.grid_size = np.asarray([50,50,50])
        self.fixed_volume_space = False

    def __len__(self):
        return self.src.shape[0]

    def getVoxel(self, xyz):
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound = np.percentile(xyz, 100, axis=0)
            min_bound = np.percentile(xyz, 0, axis=0)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        unq, unq_inv, unq_cnt = np.unique(grid_ind, return_inverse=True, return_counts=True, axis = 0)

        voxel_points = np.zeros((2048, 3))
        for t, i in enumerate(unq_inv):
            voxel_points[i] = voxel_points[i] + xyz[t] / unq_cnt[i]
        

        return  torch.from_numpy(voxel_points), torch.tensor(unq.shape[0])

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]

        label =  self.label[index]

        if self.prefix == "train":
            transform, angle_t = random_pose(self.max_angle, self.max_trans / 2)
            pose1, _ = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src_rotated = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt_rotated = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            src_rotated = self.src_rotated[index]
            tgt_rotated = self.tgt_rotated[index]
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]
            angle_t = - 1
            pose1 = torch.from_numpy(self.pose_src[index])
            pose2 = torch.from_numpy(self.pose_tgt[index])

            complete = self.complete[index]

            src  = complete

        elif self.prefix == 'test':
            src_rotated = src
            tgt_rotated = tgt

        src_rotated_vox, src_vox_len = self.getVoxel(src_rotated)
        tgt_rotated_vox, tgt_vox_len  = self.getVoxel(tgt_rotated)



        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)
        src_rotated = torch.from_numpy(src_rotated)
        tgt_rotated = torch.from_numpy(tgt_rotated)

        label = torch.from_numpy(np.array([label]))
        a,_ = translation_back(self.transforms[index,:3,:3])

        if a>45:
            add_ps = torch.ones(1)
        else:
            add_ps = torch.zeros(1)
        
        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            angle_t = torch.from_numpy(np.array([angle_t]))
            match_level = match_level
            rot_level = rot_level
            return src, tgt, src_rotated, tgt_rotated, transform, match_level, rot_level, pose1, pose2, angle_t, label, src_rotated_vox, tgt_rotated_vox, src_vox_len, tgt_vox_len, add_ps, a
        else:
            return src, tgt, label





class MVP_RG_rotated_bound(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args, l = None , r = None):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('./data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][l:r].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][l:r].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][l:r].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][l:r].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][l:r]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][l:r].astype('float32'))
                self.tgt = np.array(f['tgt'][l:r].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['src'][l:r].astype('float32'))
                self.tgt = np.array(f['tgt'][l:r].astype('float32'))
                self.src_rotated = np.array(f['rotated_src'][l:r].astype('float32'))
                self.tgt_rotated = np.array(f['rotated_tgt'][l:r].astype('float32'))
                self.transforms = np.array(f['transforms'][l:r].astype('float32'))
                self.rot_level = np.array(f['rot_level'][l:r].astype('int32'))

                self.pose_src = np.array(f['pose_src'][l:r].astype('float32'))
                self.pose_tgt = np.array(f['pose_tgt'][l:r].astype('float32'))
                self.complete =  np.array(f['complete'][l:r].astype('float32'))

                self.ori_src = np.array(f['src'][l:r].astype('float32'))
                self.ori_tgt = np.array(f['tgt'][l:r].astype('float32'))
        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]

            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)
        # from IPython import embed
        # print(self.prefix)
        # #translation_back(self.transforms[j,:3,:3])
        # embed()

    def __len__(self):
        return self.src.shape[0]


    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]

        label =  self.label[index]

        if self.prefix == "train":
            transform, angle_t = random_pose(self.max_angle, self.max_trans / 2)
            pose1, _ = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            src_rotated = src @ pose1[:3, :3].T + pose1[:3, 3]
            tgt_rotated = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            src_rotated = self.src_rotated[index]
            tgt_rotated = self.tgt_rotated[index]
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]
            angle_t = - 1
            pose1 = torch.from_numpy(self.pose_src[index])
            pose2 = torch.from_numpy(self.pose_tgt[index])

        elif self.prefix == 'test':
            src_rotated = src
            tgt_rotated = tgt


        src = torch.from_numpy(src)
        tgt = torch.from_numpy(tgt)
        src_rotated = torch.from_numpy(src_rotated)
        tgt_rotated = torch.from_numpy(tgt_rotated)

        label = torch.from_numpy(np.array([label]))
        
        if self.prefix is not "test":
            transform = torch.from_numpy(transform)
            angle_t = torch.from_numpy(np.array([angle_t]))
            match_level = match_level
            rot_level = rot_level

            a,_ = translation_back(self.transforms[index,:3,:3])
            if a>45:
                add_ps = torch.ones(1)
            else:
                add_ps = torch.zeros(1)

            return src, tgt, src_rotated, tgt_rotated, transform, label, add_ps, a
        else:
            return src, tgt, label


class MVP_RG_Aligned(Dataset):
    """docstring for MVP_RG"""
    def __init__(self, prefix, args):
        self.prefix = prefix

        if self.prefix == "train":
            f = h5py.File('./data/MVP_Train_RG.h5', 'r')
        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans

        self.label = f['cat_labels'][:].astype('int32')
        if self.prefix == "test":
            self.src = np.array(f['rotated_src'][:].astype('float32'))
            self.tgt = np.array(f['rotated_tgt'][:].astype('float32'))
        else:
            self.match_level = np.array(f['match_level'][:].astype('int32'))

            match_id = []
            for i in range(len(f['match_id'].keys())):
                ds_data = f['match_id'][str(i)][:]
                match_id.append(ds_data)
            self.match_id = np.array(match_id, dtype=object)

            if self.prefix == "train":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                if args.max_angle > 45:
                    self.rot_level = int(1)
                else:
                    self.rot_level = int(0)
            elif self.prefix == "val":
                self.src = np.array(f['src'][:].astype('float32'))
                self.tgt = np.array(f['tgt'][:].astype('float32'))
                self.src_rotated = np.array(f['rotated_src'][:].astype('float32'))
                self.tgt_rotated = np.array(f['rotated_tgt'][:].astype('float32'))
                self.transforms = np.array(f['transforms'][:].astype('float32'))
                self.rot_level = np.array(f['rot_level'][:].astype('int32'))

                self.pose_src = np.array(f['pose_src'][:].astype('float32'))
                self.pose_tgt = np.array(f['pose_tgt'][:].astype('float32'))
                self.complete =  np.array(f['complete'][:].astype('float32'))

                self.ori_src = np.array(f['src'][:].astype('float32'))
                self.ori_tgt = np.array(f['tgt'][:].astype('float32'))
        f.close()
        
        if args.category:
            self.src = self.src[self.label==args.category]
            self.tgt = self.tgt[self.label==args.category]

            if self.prefix is not "test":
                self.match_id = self.match_id[self.label==args.category]
                self.match_level = self.match_level[self.label==args.category]
                if self.prefix == False:
                    self.transforms = self.transforms[self.label==args.category]
                    self.rot_level = self.rot_level[self.label==args.category]
            self.label = self.label[self.label==args.category]

        # print(self.src.shape, self.tgt.shape, self.match_id.shape, self.match_level.shape, self.label.shape)
        # from IPython import embed
        # print(self.prefix)
        # embed()

        self.grid_size = np.asarray([50,50,50])
        self.fixed_volume_space = False

    def __len__(self):
        return self.src.shape[0]

    def getVoxel(self, xyz):
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        else:
            max_bound = np.percentile(xyz, 100, axis=0)
            min_bound = np.percentile(xyz, 0, axis=0)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
        unq, unq_inv, unq_cnt = np.unique(grid_ind, return_inverse=True, return_counts=True, axis = 0)

        #voxel_points = np.zeros((2048, 3))
        voxel_points = np.zeros((2048, 3))
        for t, i in enumerate(unq_inv):
            voxel_points[i] = voxel_points[i] + xyz[t] / unq_cnt[i]
        
        return  torch.from_numpy(voxel_points), torch.tensor(unq.shape[0]), torch.from_numpy(xyz), torch.from_numpy(grid_ind)

    def getTransform(self, transform):
        poset, _ = random_pose_lr(np.pi / 36, 0, 0.001)

        #posef, _ = random_pose_lr(np.pi / 10 / 18, np.pi * 8 / 18 , 0.001)
        posef, _ = random_pose_lr(np.pi, np.pi / 36, 0.001)
        #posef, _ = random_pose_lr(np.pi, np.pi * 16 / 18, 0.001)

        transformt = poset @ transform  
        transformf = posef @ transform

        return transformt, transformf

    def __getitem__(self, index):
        src = self.src[index]
        tgt = self.tgt[index]

        label =  self.label[index]

        if self.prefix == "train":
            transform, angle_t = random_pose(self.max_angle, self.max_trans / 2)

            transformt, transformf = self.getTransform(transform)

            pose1, _ = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            posef = transformf @ pose1
            poset = transformt @ pose1

            src_rotated = src @ pose1[:3, :3].T + pose1[:3, 3]

            src_rotated_t = src @ poset[:3, :3].T + poset[:3, 3]
            src_rotated_f = src @ posef[:3, :3].T + posef[:3, 3]
            tgt_rotated = tgt @ pose2[:3, :3].T + pose2[:3, 3]
            rot_level = self.rot_level
            match_level = self.match_level[index]

        elif self.prefix == "val":
            src_rotated = self.src_rotated[index]
            tgt_rotated = self.tgt_rotated[index]
            transform = self.transforms[index]
            rot_level = self.rot_level[index]
            match_level = self.match_level[index]
            angle_t= - 1
            pose1 = torch.from_numpy(self.pose_src[index])
            pose2 = torch.from_numpy(self.pose_tgt[index])

        elif self.prefix == 'test':
            src_rotated = src
            tgt_rotated = tgt

        src_rotated_t_vox, src_vox_t_len, xyz_t, grid_t = self.getVoxel(src_rotated_t)
        src_rotated_f_vox, src_vox_f_len, xyz_f, grid_f = self.getVoxel(src_rotated_f)
        tgt_rotated_vox, tgt_vox_len, xyz_gt, grid_gt = self.getVoxel(tgt_rotated)

        label = torch.from_numpy(np.array([label]))

        if self.prefix is not "test":
            return src_rotated_t_vox, src_rotated_f_vox, tgt_rotated_vox, src_vox_t_len, src_vox_f_len, tgt_vox_len, torch.Tensor(1), torch.Tensor(0), xyz_t, xyz_f, xyz_gt
        else:
            return src, tgt, label


class Modelnet_RG_rotated_bound(Dataset):
    """docstring for Modelnet_RG"""
    def __init__(self, prefix, args, l = None , r = None):
        self.prefix = prefix

        self.n_points = 1024

        if self.prefix == "clean":
            f = h5py.File('./data/modelnet_clearn.h5', 'r')
            self.source = f['source'][...]
            self.target = f['target'][...]
            self.transform = f['transform'][...]

        elif self.prefix == "val":
            f = h5py.File('./data/MVP_Test_RG.h5', 'r')
        elif self.prefix == "test":
            f = h5py.File('./data/MVP_ExtraTest_RG.h5', 'r')
        
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans


        from IPython import embed
        print(self.prefix)
        #translation_back(self.transforms[j,:3,:3])
        embed()

    def __len__(self):
        return self.source.shape[0]


    def __getitem__(self, index):
        src = self.source[index][:self.n_points]
        tgt = self.target[index][:self.n_points]
        transform = self.transform[index]

        return torch.from_numpy(pcd1.astype('float32')), torch.from_numpy(pcd2.astype('float32')), torch.from_numpy(transform.astype('float32'))


       

       

