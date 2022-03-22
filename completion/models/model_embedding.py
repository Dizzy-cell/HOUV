from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.modules import transformer
from torch.nn.modules.activation import ReLU
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import math

# from utils.model_utils import gen_grid_up, calc_emd, calc_cd
from model_utils import gen_grid_up, calc_emd, calc_cd

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class Embedding_Transformer(nn.Module):
    def __init__(self, nclasses = 16, embedding_size = 32, num_heads=8):
        super(Embedding_Transformer, self).__init__()

        self.embedding_size = embedding_size
        self.num_head = num_heads
        self.nclasses = nclasses

        self.trasfer1 = torch.nn.Sequential(
            nn.Conv1d(2048, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
        )

        self.trasfer2 = torch.nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
        )

        self.encoder = torch.nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )

        self.classifer = torch.nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.nclasses),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x_t1 = self.trasfer1(x)
        x_t1 = x_t1.reshape((batch_size, -1))
        x_t2 = self.trasfer2(x.transpose(1,2))
        x_t2_mean = x_t2.mean(dim = 2)
        x_t2_mx, _, = x_t2.max(dim = 2)

        x_cat = torch.cat([x_t1, x_t2_mean, x_t2_mx], dim = 1)

        embedding = self.encoder(x_cat)
        predict = self.classifer(embedding)

        return embedding, predict




class Resnet18(nn.Module):
    def __init__(self, nclasses = 16, embedding_size = 32, num_heads=8):
        super(Resnet18, self).__init__()

        self.embedding_size = embedding_size
        self.num_head = num_heads
        self.nclasses = nclasses

        self.trasfer1 = torch.nn.Sequential(
            nn.Conv1d(3, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
        )

        self.trasfer2 = torch.nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.encoder = torch.nn.Sequential(
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )

        self.classifer = torch.nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.nclasses),
        )
    
    def feature_describe(self, x, dim = 1):
        x_mx, _ = x.max(dim = dim)
        x_mi, _ = x.min(dim = dim)
        x_sz = x_mx - x_mi
        x_ret = torch.cat([x_mx, x_sz], dim = 1)
        return x_ret

    def forward(self, x):
        batch_size = x.shape[0]

        x_des = self.feature_describe(x)

        x_t1 = self.trasfer1(x.transpose(1,2))
        x_t1_des = self.feature_describe(x_t1, dim = 2)

        x_t2_des = self.trasfer2(x_des)

        x_cat = torch.cat([x_t2_des, x_t1_des], dim = 1)
        embedding = self.encoder(x_cat)
        predict = self.classifer(embedding)

        return embedding, predict


class Model(nn.Module):
    def __init__(self, args, nclasses=16):
        super(Model, self).__init__()

        self.num_points = args.num_points
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd

        
        #self.transformer = Embedding_Transformer(nclasses = nclasses)
        self.transformer = Resnet18(nclasses = nclasses)

    def forward(self,
                x,
                gt=None,
                label = None,
                prefix="train",
                mean_feature=None,
                alpha=None):

        out = self.transformer(x)

        return out