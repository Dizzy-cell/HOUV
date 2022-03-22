import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
# from utils.model_utils import *
from model_utils import *
from models.pcn import PCN_encoder, PCN_encoder_label

# proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(proj_dir, "utils/Pointnet2.PyTorch/pointnet2"))
# import pointnet2_utils as pn2

# from utils.mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation
# from ..utils import three_interpolate, furthest_point_sample, gather_points, grouping_operation
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation
from .segmentator_3d_asymm_spconv import Asymm_3d_spconv


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


class Folding(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 step_ratio,
                 global_feature_size=1024,
                 num_models=1):
        super(Folding, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.step_ratio = step_ratio
        self.num_models = num_models

        self.conv = nn.Conv1d(input_size + global_feature_size + 2,
                              output_size,
                              1,
                              bias=True)

        sqrted = int(math.sqrt(step_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (step_ratio % i) == 0:
                num_x = i
                num_y = step_ratio // i
                break

        grid_x = torch.linspace(-0.2, 0.2, steps=num_x)
        grid_y = torch.linspace(-0.2, 0.2, steps=num_y)

        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        self.grid = torch.stack([x, y], dim=-1).view(-1, 2)  # (2, 2)

    def forward(self, point_feat, global_feat):
        batch_size, num_features, num_points = point_feat.size()
        point_feat = point_feat.transpose(
            1,
            2).contiguous().unsqueeze(2).repeat(1, 1, self.step_ratio, 1).view(
                batch_size, -1, num_features).transpose(1, 2).contiguous()
        global_feat = global_feat.unsqueeze(2).repeat(
            1, 1, num_points * self.step_ratio).repeat(self.num_models, 1, 1)
        grid_feat = self.grid.unsqueeze(0).repeat(
            batch_size, num_points, 1).transpose(1, 2).contiguous().cuda()
        features = torch.cat([global_feat, point_feat, grid_feat], axis=1)
        features = F.relu(self.conv(features))
        return features


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(
            self.af(feature)))) + self.conv_res(feature)


class SK_SA_module(nn.Module):
    def __init__(self,
                 in_planes,
                 rel_planes,
                 mid_planes,
                 out_planes,
                 share_planes=8,
                 k=[10, 20],
                 r=2,
                 L=32):
        super(SK_SA_module, self).__init__()

        self.num_kernels = len(k)
        d = max(int(out_planes / r), L)

        self.sams = nn.ModuleList([])

        for i in range(len(k)):
            self.sams.append(
                SA_module(in_planes, rel_planes, mid_planes, out_planes,
                          share_planes, k[i]))

        self.fc = nn.Linear(out_planes, d)
        self.fcs = nn.ModuleList([])

        for i in range(len(k)):
            self.fcs.append(nn.Linear(d, out_planes))

        self.softmax = nn.Softmax(dim=1)
        self.af = nn.ReLU(inplace=False)

    def forward(self, input):
        x, idxs = input
        assert (self.num_kernels == len(idxs))
        for i, sam in enumerate(self.sams):
            fea, _ = sam([x, idxs[i]])
            fea = self.af(fea)
            fea = fea.unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)

        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return [fea_v, idxs]


class SKN_Res_unit(nn.Module):
    def __init__(self, input_size, output_size, k=[10, 20], layers=1):
        super(SKN_Res_unit, self).__init__()
        self.conv1 = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.sam = self._make_layer(output_size,
                                    output_size // 16,
                                    output_size // 4,
                                    output_size,
                                    int(layers),
                                    8,
                                    k=k)
        self.conv2 = nn.Conv2d(output_size, output_size, 1, bias=False)
        self.conv_res = nn.Conv2d(input_size, output_size, 1, bias=False)
        self.af = nn.ReLU(inplace=False)

    def _make_layer(self,
                    in_planes,
                    rel_planes,
                    mid_planes,
                    out_planes,
                    blocks,
                    share_planes=8,
                    k=16):
        layers = []
        for _ in range(0, blocks):
            layers.append(
                SK_SA_module(in_planes, rel_planes, mid_planes, out_planes,
                             share_planes, k))
        return nn.Sequential(*layers)

    def forward(self, feat, idx):
        x, _ = self.sam([self.conv1(feat), idx])
        x = self.conv2(self.af(x))
        return x + self.conv_res(feat)


class SA_SKN_Res_encoder(nn.Module):
    def __init__(self,
                 input_size=3,
                 k=[10, 20],
                 pk=16,
                 output_size=64,
                 layers=[2, 2, 2, 2],
                 pts_num=[3072, 1536, 768, 384]):
        super(SA_SKN_Res_encoder, self).__init__()
        self.init_channel = 64

        c1 = self.init_channel
        self.sam_res1 = SKN_Res_unit(input_size, c1, k, int(layers[0]))

        c2 = c1 * 2
        self.sam_res2 = SKN_Res_unit(c2, c2, k, int(layers[1]))

        c3 = c2 * 2
        self.sam_res3 = SKN_Res_unit(c3, c3, k, int(layers[2]))

        c4 = c3 * 2
        self.sam_res4 = SKN_Res_unit(c4, c4, k, int(layers[3]))

        self.conv5 = nn.Conv2d(c4, 1024, 1)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.conv6 = nn.Conv2d(c4 + 1024, c4, 1)
        self.conv7 = nn.Conv2d(c4 + c3, c3, 1)
        self.conv8 = nn.Conv2d(c3 + c2, c2, 1)
        self.conv9 = nn.Conv2d(c2 + c1, c1, 1)

        self.conv_out = nn.Conv2d(c1, output_size, 1)
        self.dropout = nn.Dropout()
        self.af = nn.ReLU(inplace=False)
        self.k = k
        self.pk = pk
        self.rate = 2

        self.pts_num = pts_num

    def _edge_pooling(self, features, points, rate=2, k=16, sample_num=None):
        features = features.squeeze(2)

        if sample_num is None:
            input_points_num = int(features.size()[2])
            sample_num = input_points_num // rate

        ds_features, p_idx, pn_idx, ds_points = edge_preserve_sampling(
            features, points, sample_num, k)
        ds_features = ds_features.unsqueeze(2)
        return ds_features, p_idx, pn_idx, ds_points

    def _edge_unpooling(self, features, src_pts, tgt_pts):
        features = features.squeeze(2)
        idx, weight = three_nn_upsampling(tgt_pts, src_pts)
        features = three_interpolate(features, idx, weight)
        features = features.unsqueeze(2)
        return features

    def forward(self, features):
        batch_size, _, num_points = features.size()
        pt1 = features[:, 0:3, :]

        idx1 = []
        for i in range(len(self.k)):
            idx = knn(pt1, self.k[i])
            idx1.append(idx)

        pt1 = pt1.transpose(1, 2).contiguous()

        x = features.unsqueeze(2)
        x = self.sam_res1(x, idx1)
        x1 = self.af(x)

        x, _, _, pt2 = self._edge_pooling(x1, pt1, self.rate, self.pk,
                                          self.pts_num[1])
        
        idx2 = []
        for i in range(len(self.k)):
            idx = knn(pt2.transpose(1, 2).contiguous(), self.k[i])
            idx2.append(idx)

        x = self.sam_res2(x, idx2)
        x2 = self.af(x)

        x, _, _, pt3 = self._edge_pooling(x2, pt2, self.rate, self.pk,
                                          self.pts_num[2])
        idx3 = []
        for i in range(len(self.k)):
            idx = knn(pt3.transpose(1, 2).contiguous(), self.k[i])
            idx3.append(idx)

        x = self.sam_res3(x, idx3)
        x3 = self.af(x)

        x, _, _, pt4 = self._edge_pooling(x3, pt3, self.rate, self.pk,
                                          self.pts_num[3])
        idx4 = []
        for i in range(len(self.k)):
            idx = knn(pt4.transpose(1, 2).contiguous(), self.k[i])
            idx4.append(idx)

        x = self.sam_res4(x, idx4)
        x4 = self.af(x)
        x = self.conv5(x4)
        x, _ = torch.max(x, -1)
        x = x.view(batch_size, -1)
        x = self.dropout(self.af(self.fc2(self.dropout(self.af(self.fc1(x))))))

        # from IPython import embed
        # embed()

        x = x.unsqueeze(2).repeat(1, 1, self.pts_num[3]).unsqueeze(2)
        x = self.af(self.conv6(torch.cat([x, x4], 1)))
        x = self._edge_unpooling(x, pt4, pt3)
        x = self.af(self.conv7(torch.cat([x, x3], 1)))
        x = self._edge_unpooling(x, pt3, pt2)
        x = self.af(self.conv8(torch.cat([x, x2], 1)))
        x = self._edge_unpooling(x, pt2, pt1)
        x = self.af(self.conv9(torch.cat([x, x1], 1)))
        x = self.conv_out(x)
        x = x.squeeze(2)
        return x

    def getInter(self, x, k = 1):
        x = x.squeeze(2)
        idx = knn(x, k)
        x_near = get_edge_features(x, idx)
        inter = (x_near + x.unsqueeze(2)) / 2
        return inter


class MSAP_SKN_decoder(nn.Module):
    def __init__(self,
                 num_points ,
                 num_coarse,
                 num_fine,
                 feature_size = 1024,
                 layers=[2, 2, 2, 2],
                 knn_list=[10, 20],
                 pk=10,
                 points_label=False):
        super(MSAP_SKN_decoder, self).__init__()

        self.num_points = num_points
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.points_label = points_label
        self.feature_size = feature_size

        self.generate_Coarse = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            nn.Linear(1024, num_coarse * 3),
        )

        self.dense_feature_size = 256
        self.expand_feature_size = 1024

        self.encoder = SA_SKN_Res_encoder(input_size= 3,
                                          k=knn_list,
                                          pk=pk,
                                          output_size=self.dense_feature_size,
                                          layers=layers)

        self.generate_Fine = nn.Sequential(
                nn.Conv1d(self.dense_feature_size,
                                       self.expand_feature_size, 1),
                nn.Tanh(),
                nn.Conv1d(self.expand_feature_size, 3, 1, bias=True),
        )

        self.generate_naive = nn.Sequential(
                nn.Conv1d(self.dense_feature_size,
                                       self.expand_feature_size, 1),
                nn.Tanh(),
                nn.Conv1d(self.expand_feature_size, 3, 1, bias=True),
        )


    def forward(self, global_feat, origin_points):
        batch_size = global_feat.size()[0]

        coarse_points = self.generate_Coarse(global_feat).view(batch_size, 3, self.num_coarse)

        points = torch.cat((coarse_points, origin_points), dim = 2)

        fine_feat = self.encoder(points)
        fine_points = self.generate_Fine(fine_feat)

        idx_fps = furthest_point_sample(
            fine_points.transpose(1, 2).contiguous(), self.num_points)
        features = gather_points(fine_feat, idx_fps)

        naive_points = self.generate_naive(features)

        return coarse_points, fine_points, fine_points, naive_points


class Model(nn.Module):
    def __init__(self, args, global_feature_size=1024):
        super(Model, self).__init__()

        layers = [int(i) for i in args.layers.split(',')]
        knn_list = [int(i) for i in args.knn_list.split(',')]

        self.train_loss = args.loss

        self.encoder = PCN_encoder_label(output_size=global_feature_size)

        self.spconv = Asymm_3d_spconv(args)
    
        # self.decoder = MSAP_SKN_decoder(num_points=args.num_native,
        #                                 num_fine=args.num_fine,
        #                                 num_coarse=args.num_coarse,

        #                                 feature_size = args.feature_size,

        #                                 layers=layers,
        #                                 knn_list=knn_list,
        #                                 pk=args.pk,
        #                                 points_label=args.points_label)

    def forward(self,
                x,
                gt = None,
                label = None,
                voxels = None,
                coords = None,
                prefix="train",
                mean_feature=None,
                alpha=1):

        points = x

        feat, voxel_feat = self.encoder(points, label)

        if voxels is not None:
            cat_vox_fea = []
            cat_coods = []
            for i in range(x.shape[0]):
                cat_vox_fea.append(voxel_feat[i])
                cat_coods.append(F.pad(coords[i], (1,0), 'constant', value = i))
                
            cat_vox_fea = torch.cat(cat_vox_fea, dim = 0)
            cat_coods = torch.cat(cat_coods, dim = 0)

            feat_voxel = self.spconv(cat_vox_fea, cat_coods, batch_size = x.shape[0])
        else:
            feat_voxel = torch.zeros((feat.shape[0], 1024))
        
        feat = torch.cat([feat, feat_voxel], dim = 1)

        
   
        # coarse, naive, fine, points = self.decoder(feat, points)
        # coarse = coarse.transpose(1,2).contiguous()
        # naive = naive.transpose(1,2).contiguous()
        # points = points.transpose(1,2).contiguous()
        # fine = fine.transpose(1,2).contiguous()
 
        # if prefix == 'train': 
        #     if self.train_loss == 'cd':
        #         loss1, loss1_t = calc_cd(coarse, gt)
        #         loss2, loss2_t = calc_cd(naive, gt)
        #         loss3, loss3_t = calc_cd(fine, gt)
        #         loss4, loss4_t = calc_cd(points, gt)
        #     else:
        #         raise NotImplementedError('Only CD is supported')

        #     total_train_loss = loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean()
        #     #loss_t = loss4_t.mean() + loss1_t.mean() + loss2_t.mean() + loss3_t.mean()
        #     #total_train_loss += loss_t

        #     return fine, loss4_t, total_train_loss
        # elif prefix == "val":
        #     cd_p, cd_t, f1 = calc_cd(fine, gt, calc_f1=True)
        #     return {
        #         'out1': coarse,
        #         'out2': points,
        #         'cd_p': cd_p,
        #         'cd_t': cd_t,
        #         'f1': f1
        #     }
        # elif prefix == "test":
        #     return {'result': fine}
    
#    def getVoxel(self, points, label, ):

