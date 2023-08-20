'''
the code need to be optimized 
'''
import torch
import torch.nn as nn
from typing import Iterable, Tuple
from .kmean import KMeans
from .transformer import LoFTREncoderLayer

class Hier_Matcher(nn.Module):
    def __init__(self, kp_feat_dim: int = 256)-> None:
        super().__init__()
        self.kp_feat_dim = kp_feat_dim
        self.cluster = 4
        self.att_l1 = LoFTREncoderLayer(self.kp_feat_dim, 4, 'linear')
        self.att_l2 = LoFTREncoderLayer(self.kp_feat_dim, 4, 'linear')

    def forward(
        self,
        desc2d: torch.Tensor, # N, C
        desc3d: torch.Tensor, # M, C
    )-> Tuple[torch.Tensor, torch.Tensor]:
        device = desc2d.device
        n2, n3 = len(desc2d), len(desc3d)
        full_descs = torch.cat((desc2d, desc3d), 0)
        cls, _ = KMeans(full_descs, K = self.cluster)
        full_descs = full_descs.unsqueeze(0)
        new_full_descs = torch.zeros_like(full_descs, device=device)
        for i in range(self.cluster):
            mask = (cls == i).int().unsqueeze(0)
            idxs = torch.where(mask == 1)[1]
            hier_1_descs = full_descs[:,idxs,: ]
            if len(idxs)>20:
                hier_1_descs = self.att_l1(hier_1_descs, hier_1_descs) # B, N, C
                hier_1_2_descs = self.hier_cluster(hier_1_descs, device)
                new_full_descs[:,idxs,: ] = hier_1_2_descs
            else:
                new_full_descs[:,idxs,: ] = hier_1_descs
        desc2d, desc3d= new_full_descs[:,:n2,:], new_full_descs[:,n2:,:]
        return desc2d.squeeze(), desc3d.squeeze()

    def hier_cluster(self, descs_cluster, device, hier_c = 2):
        '''
        descs_cluster: B, N, C
        '''
        descs_cluster = descs_cluster.squeeze()
        cls, _ = KMeans(descs_cluster, hier_c)
        descs_cluster = descs_cluster.unsqueeze(0)
        new_descs_cluster = torch.zeros_like(descs_cluster, device=device)
        for i in range(hier_c):
            hier_mask = (cls == i).int().unsqueeze(0)
            second_idxs = torch.where(hier_mask  == 1)[1]
            hier_2_descs = descs_cluster[:,second_idxs,: ]
            hier_2_descs = self.att_l2(hier_2_descs, hier_2_descs) # B, N, C
            new_descs_cluster[:,second_idxs,:] = hier_2_descs
        return new_descs_cluster




# hier_matcher = Hier_Matcher()
# hier_matcher.cuda()

# a = (-5 + torch.rand((200, 256))).cuda()
# b = (0 + torch.rand((200, 256))).cuda()
# c = (5 + torch.rand((200, 256))).cuda()
# d = (10 + torch.rand((200, 256))).cuda()
# e = (10 + torch.rand((200, 256))).cuda()

# f = (-5 + torch.rand((200, 256))).cuda()
# g = (0 + torch.rand((200, 256))).cuda()
# h = (5 + torch.rand((200, 256))).cuda()

# desc2d = torch.cat((a,b,c, d, e), 0)
# desc3d = torch.cat((f,g,h), 0)

# import time
# for i in range(100):
#     t0 = time.time()
#     new_desc2d, new_desc3d = hier_matcher.forward(desc2d, desc3d)
#     t1 = time.time()
#     print(t1-t0)
