import torch
import torch.nn as nn
from ..utils.extract_matches import mutual_assignment



class SC2_PCR(nn.Module):
    '''
    using the second order spatial compatibility for outlier pruning
    '''
    def __init__(self) -> None:
        super().__init__()
        self.d_thre = 0.1
        self.ratio = 1.0
        self.num_iterations = 10
        self.nms_radius=0.1

    def forward(self,
        ipts2d: torch.Tensor, # N, 2
        ipts3d: torch.Tensor, # M, 2
        ) -> torch.Tensor:
        num_pts = len(ipts2d)
        p2d = ipts2d[None, :, :] # B, N, 2
        p3d = ipts3d[None, :, :] # B, N, 2
        # compute corss dist
        p2d_dist = torch.norm((p2d[:, :, None, :] - p2d[:, None, :, :]), dim=-1)
        p3d_dist = torch.norm((p3d[:, :, None, :] - p3d[:, None, :, :]), dim=-1)
        cross_dist = torch.abs(p2d_dist - p3d_dist)
        # compute first order measure
        SC_measure = torch.clamp(1.0 - cross_dist ** 2 / self.d_thre ** 2, min=0)
        hard_SC_measure = (cross_dist < self.d_thre).float()
        # select reliable seed correspondences
        confidence = self.cal_leading_eigenvector(SC_measure, method='power')
        import pdb
        pdb.set_trace()
        seeds = self.pick_seeds(p2d_dist, confidence, R=self.nms_radius, max_num=int(num_pts * self.ratio))
        # compute second order measure
        SC2_dist_thre = self.d_thre / 2
        hard_SC_measure_tight = (cross_dist < SC2_dist_thre).float()
        seed_hard_SC_measure = hard_SC_measure.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_pts))
        seed_hard_SC_measure_tight = hard_SC_measure_tight.gather(dim=1,
                                index=seeds[:, :, None].expand(-1, -1, num_pts))
        SC2_measure = torch.matmul(seed_hard_SC_measure_tight, hard_SC_measure_tight) * seed_hard_SC_measure
        return confidence


    def cal_leading_eigenvector(self, M, method='power'):
        """
        Calculate the leading eigenvector using power iteration algorithm or torch.symeig
        Input:
            - M:      [bs, num_corr, num_corr] the compatibility matrix
            - method: select different method for calculating the learding eigenvector.
        Output:
            - solution: [bs, num_corr] leading eigenvector
        """
        if method == 'power':
            # power iteration algorithm
            leading_eig = torch.ones_like(M[:, :, 0:1])
            leading_eig_last = leading_eig
            for i in range(self.num_iterations):
                leading_eig = torch.bmm(M, leading_eig)
                leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
                if torch.allclose(leading_eig, leading_eig_last):
                    break
                leading_eig_last = leading_eig
            leading_eig = leading_eig.squeeze(-1)
            return leading_eig
        elif method == 'eig':  # cause NaN during back-prop
            e, v = torch.symeig(M, eigenvectors=True)
            leading_eig = v[:, :, -1]
            return leading_eig
        else:
            exit(-1)

    def pick_seeds(self, dists, scores, R, max_num):
        """
        Select seeding points using Non Maximum Suppression. (here we only support bs=1)
        Input:
            - dists:       [bs, num_corr, num_corr] src keypoints distance matrix
            - scores:      [bs, num_corr]     initial confidence of each correspondence
            - R:           float              radius of nms
            - max_num:     int                maximum number of returned seeds
        Output:
            - picked_seeds: [bs, num_seeds]   the index to the seeding correspondences
        """
        assert scores.shape[0] == 1

        # parallel Non Maximum Suppression (more efficient)
        score_relation = scores.T >= scores  # [num_corr, num_corr], save the relation of leading_eig
        # score_relation[dists[0] >= R] = 1  # mask out the non-neighborhood node
        score_relation = score_relation.bool() | (dists[0] >= R).bool()
        is_local_max = score_relation.min(-1)[0].float()

        score_local_max = scores * is_local_max
        sorted_score = torch.argsort(score_local_max, dim=1, descending=True)

        # max_num = scores.shape[1]

        return_idx = sorted_score[:, 0: max_num].detach()

        return return_idx





