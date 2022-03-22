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
    def __init__(self, ouput_size = 2048, nclasses = 16, embedding_size = 32, num_heads=8):
        super(Embedding_Transformer, self).__init__()

        self.trasfer1 = torch.nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
        )

        self.embedding1 = nn.Embedding(nclasses, 3 * 256)

        self.trasfer2 = torch.nn.Sequential(
            nn.Conv1d(256,512,1),
            nn.ReLU(),
            nn.Conv1d(512,128,1),
            nn.ReLU(),
        )

        self.embedding_emb = nn.Embedding(nclasses,  128 * 2048 * 10)
        #self.embedding_pos = nn.Embedding(nclasses, 3 * 2048 * 10)
        self.attention2 = torch.nn.MultiheadAttention(embed_dim=128, num_heads = num_heads)



        self.decoder = torch.nn.Sequential(
            nn.Conv1d(128,3,1),
        )
     

    def forward(self, x, label):
        batch_size = x.shape[0]

        x_t1 = self.trasfer1(x).transpose(1,2)

        x_e1 = self.embedding1(label).reshape(-1,3,256)
        y = torch.bmm(x.transpose(1,2), x_e1)

        x = torch.cat((x_t1, y), dim = 1).transpose(1,2)      # B x 4048 x 1024
        x_t2 = self.trasfer2(x).transpose(1,2)

        x_e2_emb = self.embedding_emb(label).reshape(-1, 2048 * 10, 128)
        
        out, out_atten = self.attention2(x_t2, x_e2_emb, x_e2_emb)
        
        out = self.decoder(out.transpose(1,2))
        out = out.transpose(1,2)
        return out


class PCN_encoder(nn.Module):
    def __init__(self, output_size=1024, nclasses = 16, embedding_size = 32, transform_kernel = 8):
        super(PCN_encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, output_size, 1)

        # self.embedding = nn.Embedding(nclasses, embedding_size)
        # self.transform = nn.Linear(embedding_size, 12 * transform_kernel)
        # self.transform_kernel = transform_kernel

    def forward(self, x, label):
        batch_size, _, num_points = x.size()
        x_emb = nn.Sigmoid()(self.embedding(label))
        transform = self.transform(x_emb).reshape((-1,3, 4 * self.transform_kernel))
        x_t = torch.bmm(x.transpose(1,2), transform[:,:,:3 * self.transform_kernel]) +  transform[:,:,3 * self.transform_kernel:].reshape((-1, 1, 3 * self.transform_kernel))
        x_t = x_t.reshape(-1, 2048, 3, self.transform_kernel)
        x_t = x_t.transpose(1,2).reshape(-1,3, 2048 * self.transform_kernel)

        x = torch.cat((x,x_t), dim = 2)

        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature, _ = torch.max(x, 2)
        x = torch.cat((x, global_feature.view(batch_size, -1, 1).repeat(
            1, 1, num_points * (1 + self.transform_kernel)).contiguous()), 1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature, _ = torch.max(x, 2)
        return global_feature.view(batch_size, -1)


class PCN_decoder(nn.Module):
    def __init__(self, num_coarse, num_fine, scale, cat_feature_num, nclasses = 16, embedding_size = 32):
        super(PCN_decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_coarse * 3)

        self.scale = scale
        self.grid = gen_grid_up(2**(int(math.log2(scale))),
                                0.05).cuda().contiguous()
        self.conv1 = nn.Conv1d(cat_feature_num, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

        # self.embedding1 = nn.Embedding(nclasses, embedding_size)
        # self.MultiheadAttention1 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads = self.num_heads)

        # self.embedding2 = nn.Embedding(nclasses, embedding_size)
        # self.MultiheadAttention2 = nn.MultiheadAttention(embed_dim=embedding_size, num_heads = self.num_heads)


    def forward(self, x, label):
        batch_size = x.size()[0]
        
        # x_e1 = self.embedding1(label)
        # x_a1 = self.MultiheadAttention(x, x_e1, x_e1)
        # x = F.relu(self.fc1(x))

        # x_e2 = self.embedding2(label)
        # x_a2 = self.MultiheadAttention2(x, x_e2, x_e2)
        # x = F.relu(self.fc2(x))


        coarse = F.relu(self.fc1(x))
        coarse = F.relu(self.fc2(coarse))
        coarse = self.fc3(coarse).view(-1, 3, self.num_coarse)

        grid = self.grid.clone().detach()
        grid_feat = grid.unsqueeze(0).repeat(
            batch_size, 1, self.num_coarse).contiguous().cuda()

        point_feat = ((coarse.transpose(
            1, 2).contiguous()).unsqueeze(2).repeat(1, 1, self.scale, 1).view(
                -1, self.num_fine, 3)).transpose(1, 2).contiguous()

        global_feat = x.unsqueeze(2).repeat(1, 1, self.num_fine)

        feat = torch.cat((grid_feat, point_feat, global_feat), 1)

        center = ((coarse.transpose(1, 2).contiguous()).unsqueeze(2).repeat(
            1, 1, self.scale, 1).view(-1, self.num_fine,
                                      3)).transpose(1, 2).contiguous()

        fine = self.conv3(F.relu(self.conv2(F.relu(
            self.conv1(feat))))) + center
        return coarse, fine


class Model(nn.Module):
    def __init__(self, args, num_coarse=1024):
        super(Model, self).__init__()

        self.num_coarse = num_coarse
        self.num_points = args.num_points
        self.train_loss = args.loss
        self.eval_emd = args.eval_emd
        self.scale = self.num_points // num_coarse
        self.cat_feature_num = 2 + 3 + 1024

        # self.encoder = PCN_encoder()
        # self.decoder = PCN_decoder(num_coarse, self.num_points, self.scale,
        #                            self.cat_feature_num)
        
        self.transformer = Embedding_Transformer()

    def forward(self,
                x,
                gt=None,
                label = None,
                prefix="train",
                mean_feature=None,
                alpha=None):

        out = self.transformer(x, label)
        out1 = out
        out2 = out

        # feat = self.encoder(x, label)
        # out1, out2 = self.decoder(feat, label)
        # out1 = out1.transpose(1, 2).contiguous()
        # out2 = out2.transpose(1, 2).contiguous()

        if prefix == "train":
            if self.train_loss == 'emd':
                loss1 = calc_emd(out1, gt)
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1, _ = calc_cd(out1, gt)
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss1.mean() + loss2.mean() * alpha
            return out2, loss2, total_train_loss
        elif prefix == "val":
            if self.eval_emd:
                emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            else:
                emd = 0
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {
                'out1': out1,
                'out2': out2,
                'cd_p': cd_p,
                'cd_t': cd_t,
                'f1': f1
            }
        else:
            return {'result': out2}