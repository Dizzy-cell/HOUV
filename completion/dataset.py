from functools import partial
from re import I
from numpy.core.fromnumeric import nonzero
import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)
        else:
            self.gt_data = self.input_data
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

        self.scale = True

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        label = (self.labels[index])

        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // 26]))

            if self.scale:
                alpha = random.randint(8,12) * 1.0 / 10
                partial = partial * alpha
                complete = complete * alpha

            return label, partial, complete
        else:
            return label[0], partial, partial


class MVP_CP_EX(data.Dataset):
    def __init__(self, prefix="train", grid_size = [50,50,50], 
                 fixed_volume_space= True, max_volume_space=[0.5, 0.5, 0.5], min_volume_space=[-0.5, -0.5, -0.5],
                 scale_aug=True):
        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)
        else:
            self.gt_data = self.input_data
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)

        input_file.close()
        self.len = self.input_data.shape[0]

        self.scale = scale_aug
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        label = (self.labels[index])


        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // 26]))

            if self.scale:
                alpha = random.randint(8,12) * 1.0 / 10
                partial = partial * alpha
                complete = complete * alpha
            
            voxels, coords =  self.point_to_voxel(partial.numpy())
            return label, partial, complete, voxels, coords
        else:
            voxels, coords =  self.point_to_voxel(partial.numpy())
            return label[0], partial, partial, voxels, coords

    def point_to_voxel(self, xyz):

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

            return torch.from_numpy(xyz), torch.from_numpy(grid_ind)




class MVP_CP_voxel_point(data.Dataset):
    def __init__(self, prefix="train", grid_size = [50,50,50], 
                 fixed_volume_space= True, max_volume_space=[0.5, 0.5, 0.5], min_volume_space=[-0.5, -0.5, -0.5],
                 rotate_aug=False, flip_aug=False, 
                 scale_aug=False,
                 transform_aug=False):

        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix
        self.grid_size = np.asarray(grid_size)
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def point_to_voxel(self, xyz):

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

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        data_tuple = (torch.from_numpy(voxel_position),) # processed_label), return voxel_label is not necessary! 

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz =  - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        return_xyz = np.insert(return_xyz, 0, 1, axis = 1)
        data_tuple += (torch.from_numpy(grid_ind), torch.from_numpy(return_xyz))

        return data_tuple


    def __getitem__(self, index):
        partial = self.input_data[index]

        data_partial = self.point_to_voxel(partial)

        if self.prefix is not "test":
            complete = self.gt_data[index // 26]
            label = (self.labels[index])

            data_complete = self.point_to_voxel(complete)

            return label, data_partial, data_complete
        else:

            return data_partial
        

class MVP_CP_choose(data.Dataset):
    def __init__(self, prefix="train", idx = 0):
        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.idx = idx
        self.labels = np.array(input_file['labels'][()])

        self.choose = (self.labels == self.idx)
        if prefix == 'test':
            self.choose = self.choose.reshape(-1)
        self.input_choose = self.input_data[self.choose]
        self.nonzero = self.choose.nonzero()[0]
        
        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_choose.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_choose[index]))

        if self.prefix is not "test":
            complete = torch.from_numpy(self.gt_data[self.nonzero[index] // 26])
            label = self.idx
            return label, partial, complete
        else:
            return -1, partial, partial
    
    def getData(self, st):
        ans  = self.input_choose[st]
        if self.prefix is not "test":
            gt_list = self.gt_data[self.nonzero[st] // 26]
        else:
            gt_list = ans
        return ans, gt_list
    
    def rangeMinMax(self):
        xyz_mx = np.max(self.input_choose, axis = (0,1)) 
        xyz_mi = np.min(self.input_choose, axis = (0,1))
        return xyz_mi, xyz_mx

class Voxel(data.Dataset):
    def __init__(self, dataset, grid_size = np.asarray([50,50,50]), fixed_volume_space = True, max_volume_space = np.asarray([0.5,0.5,0.5]), min_volume_space = np.asarray([-0.5,-0.5,-0.5])):
        
        self.input_data = dataset
        self.grid_size = grid_size
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
    
    def __len__(self):
        return self.dataset.shape[0]

    def point_to_voxel(self, xyz):
        if self.fixed_volume_space:
            max_bound = self.max_volume_space
            min_bound = self.min_volume_space
        else:
            max_bound = np.percentile(xyz, 100, axis=0)
            min_bound = np.percentile(xyz, 0, axis=0)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        data_tuple = (torch.from_numpy(voxel_position),) # processed_label), return voxel_label is not necessary! 

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz =  - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        return_xyz = np.insert(return_xyz, 0, 1, axis = 1)
        data_tuple += (torch.from_numpy(grid_ind), torch.from_numpy(return_xyz))

        return data_tuple
    
    def __getitem__(self, index):
        partial = self.input_data[index]
        voxel_partial = self.point_to_voxel(partial)

        return voxel_partial


class MVP_CP_choose_triple(data.Dataset):
    def __init__(self, prefix="train", idx = 0):
        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.idx = idx
        self.labels = np.array(input_file['labels'][()])

        self.choose = (self.labels == self.idx)
        self.input_choose = self.input_data[self.choose]
        self.nonzero = self.choose.nonzero()[0]
        
        print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_choose.shape[0]

        self.ran_i = 1
        self.ran_j = 30

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_choose[index]))
        
        nerb = (index // 26) * 26 + self.ran_i
        self.ran_i = (self.ran_i + index) % 26

        partial_nerb = torch.from_numpy((self.input_choose[nerb]))

        other = (index + self.ran_j) %  self.len
        self.ran_j = (self.ran_j + self.ran_i * index) % (self.len - 26) + 26

        partial_other = torch.from_numpy((self.input_choose[other]))

        if self.prefix is not "test":
            complete = torch.from_numpy(self.gt_data[self.nonzero[index] // 26])
            label = self.idx
            return label, partial, partial_nerb, partial_other, complete, index // 26
        else:
            return -1, partial, partial, partial, partial, index // 26
    
    def getData(self, st):
        ans  = self.input_choose[st]
        if self.prefix is not "test":
            gt_list = self.gt_data[self.nonzero[st] // 26]
        else:
            gt_list = ans
        return ans, gt_list
    
    def rangeMinMax(self):
        xyz_mx = np.max(self.input_choose, axis = (0,1)) 
        xyz_mi = np.min(self.input_choose, axis = (0,1))
        return xyz_mi, xyz_mx