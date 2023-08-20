import torch
import torch.nn.functional as F
import torch.nn as nn
from .point_resnet import SinusoidalPositionalEmbedding
from einops import rearrange
from typing import Iterable, List, Sequence, Tuple


def cluster_encoding(cls, idesc, K = 10):
    '''get the descs of each cluster based on the points encoding '''
    device = idesc.device
    cluster_enc = torch.zeros(K, idesc.shape[1]).to(device)
    for i in range(K):
        idx = torch.where(cls == i)[0]
        if len(idx) != 0:
            cluster_descs = idesc[idx, :]
            mean_features = torch.mean(cluster_descs, dim=0)
            cluster_enc[i, :] = mean_features
    return cluster_enc

def get_embedding_angular(coords, knn_indices, angular_factor):
    r"""Compute the indices triplet-wise angular embedding.
    Args:
        coords: torch.Tensor (B, 2, N)
        knn_indices: torch.Tensor (B, N, K)
        angular_factor: scalar in [0, 1]
    Returns:
        a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
    """
    coords = coords.transpose(1, 2)
    b, n, k = knn_indices.shape
    knn_indices = knn_indices.unsqueeze(3).expand(b, n, k, 2)  # (B, N, k, 2)
    expanded_coords = coords.unsqueeze(1).expand(b, n, n, 2)  # (B, N, N, 2)
    knn_coords = torch.gather(expanded_coords, dim=2, index=knn_indices)  # (B, N, k, 2)
    ref_vectors = knn_coords - coords.unsqueeze(2)  # (B, N, k, 3)
    anc_vectors = coords.unsqueeze(1) - coords.unsqueeze(2)  # (B, N, N, 2)
    ref_vectors = ref_vectors.unsqueeze(2).expand(b, n, n, k, 2)  # (B, N, N, k, 2)
    anc_vectors = anc_vectors.unsqueeze(3).expand(b, n, n, k, 2)  # (B, N, N, k, 2)

    dot_product = torch.sum(ref_vectors * anc_vectors, dim=-1)
    magnitude_ref = torch.norm(ref_vectors, dim=-1)
    magnitude_anc = torch.norm(anc_vectors, dim=-1)
    angles = torch.atan2(dot_product, magnitude_ref * magnitude_anc)
    a_indices = angles * angular_factor
    return a_indices

def square_distance(
    src: torch.Tensor, dst: torch.Tensor, normalized: bool = False
) -> torch.Tensor:
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalized:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def get_graph_feature(
    coords: torch.Tensor, feats: torch.Tensor, k: int = 10, get_ang = False
) -> torch.Tensor:
    """
    Apply KNN search based on coordinates, then concatenate the features to the centroid features
    Input:
        X:          [B, 3, N]
        feats:      [B, C, N]
    Return:
        feats_cat:  [B, 2C, N, k]
    """
    # apply KNN search to build neighborhood
    B, C, N = feats.size()
    k = min(k, N - 1)  # There are cases the input data points are fewer than k
    dist = square_distance(coords.transpose(1, 2), coords.transpose(1, 2))
    idx = dist.topk(k=k + 1, dim=-1, largest=False, sorted=True)[
        1
    ]  # [B, N, K+1], here we ignore the smallest element as it's the query itself
    idx = idx[:, :, 1:]  # [B, N, K]

    # get triplet-wise angular embedding with top k
    if get_ang:
        a_embeding = get_embedding_angular(coords, idx, 0.5)


    idx = idx.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, N, K]
    all_feats = feats.unsqueeze(2).repeat(1, 1, N, 1)  # [B, C, N, N]

    neighbor_feats = torch.gather(all_feats, dim=-1, index=idx)  # [B, C, N, K]

    # concatenate the features with centroid
    feats = feats.unsqueeze(-1).repeat(1, 1, 1, k)

    feats_cat = torch.cat((feats, neighbor_feats - feats), dim=1)

    if get_ang:
        return feats_cat, a_embeding
    else:
        return feats_cat

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)
        self.proj_p = nn.Linear(self.d_model, self.d_model)

        # self.dropout = build_dropout_layer(dropout)

    def forward(self, input_q, input_k, input_v, embed_qk, key_weights=None, key_masks=None, attention_factors=None):
        r"""Scaled Dot-Product Attention with Pre-computed Relative Positional Embedding (forward)
        Args:
            input_q: torch.Tensor (B, N, C)
            input_k: torch.Tensor (B, M, C)
            input_v: torch.Tensor (B, M, C)
            embed_qk: torch.Tensor (B, N, M, C), relative positional embedding
            key_weights: torch.Tensor (B, M), soft masks for the keys
            key_masks: torch.Tensor (B, M), True if ignored, False if preserved
            attention_factors: torch.Tensor (B, N, M)
        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: torch.Tensor (B, H, N, M)
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        p = rearrange(self.proj_p(embed_qk), 'b n m (h c) -> b h n m c', h=self.num_heads)

        attention_scores_p = torch.einsum('bhnc,bhnmc->bhnm', q, p)
        attention_scores_e = torch.einsum('bhnc,bhmc->bhnm', q, k)
        attention_scores = (attention_scores_e + attention_scores_p) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        # attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states

class SelfAttention(nn.Module):
    def __init__(self, feature_dim: int, k: int = 10) -> None:
        super(SelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(feature_dim)

        self.conv2 = nn.Conv2d(
            feature_dim * 2, feature_dim * 2, kernel_size=1, bias=False
        )
        self.in2 = nn.InstanceNorm2d(feature_dim * 2)

        self.conv3 = nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1, bias=False)
        self.in3 = nn.InstanceNorm2d(feature_dim)
        self.ang_embedding = SinusoidalPositionalEmbedding(feature_dim)
        self.proj1 = nn.Linear(feature_dim, feature_dim)
        self.multiheadatt = MultiHeadAttention(d_model=feature_dim, num_heads=4)
        self.proj2 = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

        self.k = k

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Here we take coordinats and features, feature aggregation are guided by coordinates
        Input:
            coords:     [B, 3, N]
            feats:      [B, C, N]
        Output:
            feats:      [B, C, N]
        """
        B, C, N = features.size()

        x0 = features.unsqueeze(-1)  # [B, C, N, 1]

        x1, a1 = get_graph_feature(coords, x0.squeeze(-1), self.k, get_ang = True)
        x1 = F.leaky_relu(self.in1(self.conv1(x1)), negative_slope=0.2)
        x1 = x1.max(dim=-1, keepdim=True)[0]
        a1 = self.ang_embedding(a1)
        a1 = self.proj1(a1)
        a1 = a1.mean(dim=3) # [B, N, N, K]

        x2 = get_graph_feature(coords, x1.squeeze(-1), self.k, get_ang = False)
        x2 = F.leaky_relu(self.in2(self.conv2(x2)), negative_slope=0.2)
        x2 = x2.max(dim=-1, keepdim=True)[0]

        x3 = torch.cat((x0, x1, x2), dim=1)
        x3 = F.leaky_relu(self.in3(self.conv3(x3)), negative_slope=0.2).view(B, -1, N)

        # encoding angular informattion
        x3 = x3.permute(0, 2, 1)
        x4 = self.multiheadatt(x3, x3, x3, a1)
        x4 = self.proj2(x4)
        x4 = self.norm(x3 + x4)
        # scores = torch.einsum('bnc,bnmc->bnm', x3, a1)
        # scores = F.softmax(scores, dim=-1)
        # x4 = torch.matmul(scores, x3)
        return x4.permute(0,2,1)

class Coarse_gcn_encoding(nn.Module):
    '''
    Predator based embedding
    '''
    def __init__(self, feature_dim: int = 128, k: int = 4,) -> None:
        super().__init__()
        self.layer = SelfAttention(feature_dim, k)

    def forward(
        self,
        desc0: torch.Tensor,
        desc1: torch.Tensor,
        coords0: torch.Tensor,
        coords1: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        coords0 = coords0[None,:,:].permute(0,2,1)
        coords1 = coords1[None,:,:].permute(0,2,1)
        desc0 = desc0[None,:,:].permute(0,2,1)
        desc1 = desc1[None,:,:].permute(0,2,1)
        desc0 = self.layer(coords0, desc0)
        desc1 = self.layer(coords1, desc1)
        return desc0.permute(0,2,1).squeeze(), desc1.permute(0,2,1).squeeze()
