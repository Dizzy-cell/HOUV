#from MVP_Benchmark.MVP_Benchmark.completion.models.embedding_pcn import PCN_decoder
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from utils.model_utils import *
from model_utils import *
from models.pcn import PCN_encoder

# proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
# import pointnet2_utils as pn2

# from utils.mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation
# from ..utils import three_interpolate, furthest_point_sample, gather_points, grouping_operation
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation


class SA_module(nn.Module):
    def __init__(self,
                 in_planes,
                 rel_planes,
                 mid_planes,
                 out_planes,
                 share_planes=8,
                 k=16):
        super(SA_module, self).__init__()
        self.share_planes = share_planes
        self.k = k
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, mid_planes, kernel_size=1)

        self.conv_w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(rel_planes * (k + 1),
                      mid_planes // share_planes,
                      kernel_size=1,
                      bias=False), nn.ReLU(inplace=False),
            nn.Conv2d(mid_planes // share_planes,
                      k * mid_planes // share_planes,
                      kernel_size=1))
        self.activation_fn = nn.ReLU(inplace=False)

        self.conv_out = nn.Conv2d(mid_planes, out_planes, kernel_size=1)

    def forward(self, input):
        x, idx = input
        batch_size, _, _, num_points = x.size()
        identity = x  # B C 1 N
        x = self.activation_fn(x)
        xn = get_edge_features(x, idx)  # B C K N
        x1, x2, x3 = self.conv1(x), self.conv2(xn), self.conv3(xn)

        x2 = x2.view(batch_size, -1, 1, num_points).contiguous()  # B kC 1 N
        w = self.conv_w(torch.cat([x1, x2], 1)).view(batch_size, -1, self.k,
                                                     num_points)
        w = w.repeat(1, self.share_planes, 1, 1)
        out = w * x3
        out = torch.sum(out, dim=2, keepdim=True)

        out = self.activation_fn(out)
        out = self.conv_out(out)  # B C 1 N
        out += identity
        return [out, idx]


class AttentionUp(nn.Module):
    def __init__(self, input_size = (32, 2048), output_size = (64, 4056), num_heads = 8):
        super(AttentionUp, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        embedding_size = output_size[0]

        self.fc = nn.Conv1d(input_size[0], output_size[0], 1)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads = num_heads)
    
    def forward(self, x):
        x1 = F.relu(self.fc(x))
        x1_t = x1.transpose(1,2)
        x2, _ = self.attention(x1_t, x1_t, x1_t)
        x2 = x2.transpose(1,2)
        x = torch.cat([x2, x1], dim = 2)
        return x, x2

class AttentionDown(nn.Module):
    def __init__(self, input_size = (64, 4056), output_size = (32, 2048), num_heads = 4):
        super(AttentionDown, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        embedding_size = output_size[0]

        self.fc1 = nn.Conv1d(input_size[0], output_size[0], 1)
        self.fc2 = nn.Conv1d(input_size[0], output_size[0], 1)
        self.fc3 = nn.Conv1d(input_size[0], output_size[0], 1)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size, num_heads = num_heads)
    
    def forward(self, x, a):
        size = a.shape[2]
        x1, x2 = x[:,:,:size], x[:,:,size:]
        
        x1 = F.relu(self.fc1(x1))
        x1_t = x1.transpose(1,2)
        x1, _ = self.attention(x1_t, x1_t, x1_t)
        x1 = x1.transpose(1,2)

        a = F.relu(self.fc2(a))
        x2 = F.relu(self.fc3(x2))

        x2 = x2 * a
        x = x1 + x2

        return x



class Encoder(nn.Module):
    def __init__(self, output_size=1024):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.conv3 = nn.Conv1d(1024, 1024, 1)
        self.conv4 = nn.Conv1d(1024, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(
            1, 1, num_points).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size = 1024, output_size=3):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.conv3 = nn.Conv1d(1024, output_size, 1)

    def forward(self, x):
        batch_size, _, num_points = x.size()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(
            self.af(feature)))) + self.conv_res(feature)
        
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.num_pt = 2048
        self.channel = 32

        self.embedding = nn.Embedding(16, 128)
        self.encoder = Encoder(output_size = 1024)

  
        self.decoder = Decoder(input_size = 1024 + 128 , output_size = 3)
        encoder_layer = nn.TransformerEncoderLayer(d_model= 1024 + 128, nhead=4)
        self.transformEncoder = nn.TransformerEncoder(encoder_layer, num_layers=1)



    def forward(self, x, label = None):

        x = self.encoder(x)
        if label is not None:
            x_label = self.embedding(label)
            x_label = x_label.unsqueeze(2).repeat(1, 1, x.shape[2])
            x = torch.cat([x,x_label], dim = 1)
        else:
            x_label = torch.zeros((x.shape[0], 128, x.shape[2]))
            x = torch.cat([x, x_label], dim = 1)

        x = x.transpose(1,2).transpose(0,1).contiguous()
        x = self.transformEncoder(x)
        x = x.transpose(0,1).transpose(1,2).contiguous()

        x = self.decoder(x)

        return x
     

class Model(nn.Module):
    def __init__(self, args, size_z=128, global_feature_size=1024):
        super(Model, self).__init__()

        self.transformer = Transformer(args)
      
    def forward(self, x, gt = None, label = None, prefix = 'train'):

        x = self.transformer(x, label = label)
        x = x.transpose(1,2).contiguous()

        if prefix == "train":

                cd_p, cd_t = calc_cd(x, gt)
          
                total_train_loss = cd_p.mean() + cd_t.mean() * 0.02
                return x, cd_t, total_train_loss
        elif prefix == "val":
            cd_p, cd_t, f1 = calc_cd(x, gt, calc_f1=True)
            return {
                'out1': x,
                'out2': x,
                #'emd': emd,
                'cd_p': cd_p,
                'cd_t': cd_t,
                'f1': f1
            }
        else:
            return {'result': x}