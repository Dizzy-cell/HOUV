# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=3,
                             stride=stride,
                             padding=1,
                             bias=False,
                             indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(1, 3, 3),
                             stride=stride,
                             padding=(0, 1, 1),
                             bias=False,
                             indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(1, 1, 3),
                             stride=stride,
                             padding=(0, 0, 1),
                             bias=False,
                             indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(1, 3, 1),
                             stride=stride,
                             padding=(0, 1, 0),
                             bias=False,
                             indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(3, 1, 1),
                             stride=stride,
                             padding=(1, 0, 0),
                             bias=False,
                             indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=(3, 1, 3),
                             stride=stride,
                             padding=(1, 0, 1),
                             bias=False,
                             indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes,
                             out_planes,
                             kernel_size=1,
                             stride=stride,
                             padding=1,
                             bias=False,
                             indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 stride=1,
                 indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters,
                               out_filters,
                               indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        resA.features = resA.features + shortcut.features

        return resA


class ResBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 stride=1,
                 pooling=True,
                 drop_out=True,
                 height_pooling=True,
                 indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv1x3(out_filters,
                               out_filters,
                               indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv1x3(in_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv3x1(out_filters,
                             out_filters,
                             indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters,
                                                out_filters,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                indice_key=indice_key,
                                                bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters,
                                                out_filters,
                                                kernel_size=3,
                                                stride=(2, 2, 1),
                                                padding=1,
                                                indice_key=indice_key,
                                                bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)

        resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 indice_key=None,
                 up_key=None):
        super(UpBlock, self).__init__()

        self.trans_dilao = conv3x3(in_filters,
                                   out_filters,
                                   indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        #self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()

        self.up_subm = spconv.SparseInverseConv3d(out_filters,
                                                  out_filters,
                                                  kernel_size=3,
                                                  indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.features = self.trans_act(upA.features)

        ## upsample
        upA = self.up_subm(upA)

        upA.features = upA.features + skip.features

        upE = self.conv1(upA)
        upE.features = self.act1(upE.features)

        upE = self.conv2(upE)
        upE.features = self.act2(upE.features)

        upE = self.conv3(upE)
        upE.features = self.act3(upE.features)

        return upE


class ReconBlock(nn.Module):
    def __init__(self,
                 in_filters,
                 out_filters,
                 kernel_size=(3, 3, 3),
                 stride=1,
                 indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters,
                               out_filters,
                               indice_key=indice_key + "bef")

        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters,
                                 out_filters,
                                 indice_key=indice_key + "bef")

        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters,
                                 out_filters,
                                 indice_key=indice_key + "bef")

        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.act1_2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 args,
                 input_shape = [50,50,50],
                 num_input_features=1024,
                 nclasses=16,
                 n_height=32,
                 init_size=128):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(input_shape)
   
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.init_size = init_size
        self.downCntx = ResContextBlock(num_input_features,
                                        init_size,
                                        indice_key="pre")
        self.resBlock2 = ResBlock(4 * init_size,
                                  4 * init_size,
                                  height_pooling=True,
                                  indice_key="down2")
        self.resBlock3 = ResBlock(4 * init_size,
                                  4 * init_size,
                                  height_pooling=True,
                                  indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size,
                                  4 * init_size,
                                  height_pooling=True,
                                  indice_key="down4")

        self.ReconNet = ReconBlock(4 * init_size,
                                   4 * init_size,
                                   indice_key="recon")


    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        fea = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)

        fea = self.downCntx(fea)
        down1c, down1b = self.resBlock2(fea)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)

        down4c = self.ReconNet(down3c)

        a = (down3c.dense()).reshape((batch_size, 4 * self.init_size, -1))
        b = (down4c.dense()).reshape(((batch_size, 4 * self.init_size, -1)))

        features = torch.cat([a,b], dim = 1)
        features, _= torch.max(features, dim = 2)

        return features
