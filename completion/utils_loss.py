import torch
import torch_scatter
import torch.nn.functional as F
import spconv

def getGt(pt_fea, xy_ind, sparse_shape = [50,50,50]):
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

    voxel_features = torch_scatter.scatter_mean(cat_pt_fea,
                                                 unq_inv,
                                                 dim=0)

    ret = spconv.SparseConvTensor(voxel_features, unq, sparse_shape,
                                      pt_fea.shape[0])

    points = pt_fea[:,:,1:4]   # B x N x 3
    ret = ret.dense()

    one_hot = ret[:,0,:,:,:].type(torch.LongTensor).to(pt_fea.device)  # B x 1 X 50 x 50 x 50
    re_pos = ret[:,1:,:,:,:].type(torch.FloatTensor).to(pt_fea.device)  # B x 1 x 50 x 50 x 50

    return one_hot, re_pos, points


def getPts_2048(vox_predict, vox_position, vox_bias, k = 2048, spare_shape = [50,50,50]):
    a = vox_predict[:,1].reshape(vox_predict.shape[0], -1)
    b = vox_position.reshape(vox_position.shape[0], vox_position.shape[1], -1).transpose(1, 2)
    values, indices = torch.topk(a, k, dim = 1)
    dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), b.size(2))
    pts = torch.gather(b, 1, dummy)
    return pts

# dummy = B.unsqueeze(2).expand(B.size(0), B.size(1), A.size(2)) out = torch.gather(A, 1, dummy)
# dummy = indices.unsqueeze(2).expand(indices.size(0), indices.size(1), b.size(2))
# out = torch.gather(b, 1, dummy)


def getPts_50(vox_predict, vox_fea, spare_shape = [50,50,50]):
    vox_predict = torch.argmax(vox_predict, dim = 1)
    pts = 0
    return pts