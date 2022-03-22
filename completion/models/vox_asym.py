from __future__ import print_function
from IPython.terminal.embed import embed
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math

from model_utils import gen_grid_up, calc_emd, calc_cd
import torch_scatter
from .segmentator_3d_asymm_spconv import Asymm_3d_spconv


class vox_fea(nn.Module):
    def __init__(self,
                 grid_size,
                 fea_dim=3,
                 out_pt_fea_dim=64,
                 max_pt_per_encode=64,
                 fea_compre=None):
        super(vox_fea, self).__init__()

        self.PPmodel = nn.Sequential(nn.BatchNorm1d(fea_dim),
                                     nn.Linear(fea_dim, 64),
                                     nn.BatchNorm1d(64), nn.ReLU(),
                                     nn.Linear(64, 128), nn.BatchNorm1d(128),
                                     nn.ReLU(), nn.Linear(128, 256),
                                     nn.BatchNorm1d(256), nn.ReLU(),
                                     nn.Linear(256, out_pt_fea_dim))

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size,
                                                stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre), nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):
        # concate everything
        cat_pt_ind = []
        for i_batch in range(xy_ind.shape[0]):
            cat_pt_ind.append(
                F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        cat_pt_fea = pt_fea.reshape((-1, pt_fea.shape[-1]))
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,
                                             return_inverse=True,
                                             return_counts=True,
                                             dim=0)              
        unq = unq.type(torch.int64)

        # process feature
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_mean(processed_cat_pt_fea,
                                                 unq_inv,
                                                 dim=0)
        # pooled_data = torch_scatter.scatter_mean(processed_cat_pt_fea,
        #                                     unq_inv,
        #                                     dim=0)

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data


class vox_asym(nn.Module):
    def __init__(
        self,
        vox_encoder,
        segmentator_spconv,
        sparse_shape,
    ):
        super().__init__()
        self.name = "vox_asym"

        self.vox_3d_generator = vox_encoder

        self.vox_3d_spconv_seg = segmentator_spconv

        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten):
        batch_size = train_pt_fea_ten.shape[0]
        coords, features_3d = self.vox_3d_generator(train_pt_fea_ten,
                                                    train_vox_ten)

        spatial_features, vox_features = self.vox_3d_spconv_seg(
            features_3d, coords, batch_size)

        out_predict = spatial_features[:,:2]
        out_pos = spatial_features[:,2:]
        
        out_predict = torch.nn.functional.softmax(out_predict)

        return out_predict, out_pos


def build_model(args):
    vox_3d_generator = vox_fea(grid_size=args.grid_size,
                               fea_dim=args.fea_dim,
                               out_pt_fea_dim=args.embedding_dim)

    vox_3d_spconv_seg = Asymm_3d_spconv(output_shape=args.grid_size,
                                        num_input_features=args.embedding_dim,
                                        init_size=args.init_size,
                                        nclasses=args.output_shape)
    model = vox_asym(vox_encoder=vox_3d_generator,
                     segmentator_spconv=vox_3d_spconv_seg,
                     sparse_shape=args.grid_size)

    return model