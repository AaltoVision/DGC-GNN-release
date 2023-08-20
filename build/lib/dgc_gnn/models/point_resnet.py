from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np


def conv1d_layer(
    in_channel: int, out_channel: int, normalize: Optional[str] = "ins"
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Conv1d(in_channel, out_channel, kernel_size=1)]
    if normalize == "ins":
        layers.append(nn.InstanceNorm1d(in_channel))
    if normalize and "bn" in normalize:
        layers.append(
            nn.BatchNorm1d(in_channel, track_running_stats=(normalize == "bn_untrack"))
        )
    return nn.Sequential(*layers)


class conv1d_residual_block(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        mid_channel: Optional[int] = None,
        normalize: Optional[str] = "ins",
        activation: str = "relu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        mid_channel = out_channel if mid_channel is None else mid_channel
        self.preconv = conv1d_layer(
            in_channel=in_channel, out_channel=mid_channel, normalize=None
        )
        self.conv1 = conv1d_layer(
            in_channel=mid_channel, out_channel=mid_channel, normalize=normalize
        )
        self.conv2 = conv1d_layer(
            in_channel=mid_channel, out_channel=out_channel, normalize=normalize
        )
        self.act = nn.LeakyReLU() if "leaky" in activation else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_residual = x
        x = self.preconv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        if self.residual:
            x = x + x_residual
        return x


class Onehot_mlp(nn.Module):
    def __init__(
        self,
        in_channel: int = 40,
        hidden_dim: int = 128,
        out_dim = 256
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channel, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x[None,:,:]

class PointResNet(nn.Module):
    def __init__(
        self,
        in_channel: int,
        num_layers: int = 12,
        feat_channel: int = 128,
        mid_channel: int = 128,
        activation: str = "relu",
        normalize: Optional[str] = "ins",
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        # First convolution
        self.conv_in = nn.Sequential(
            *[nn.Conv1d(in_channel, feat_channel, kernel_size=1)]
        )
        for i in range(self.num_layers):
            setattr(
                self,
                f"conv_{i}",
                conv1d_residual_block(
                    in_channel=feat_channel,
                    out_channel=feat_channel,
                    mid_channel=mid_channel,
                    normalize=normalize,
                    activation=activation,
                    residual=residual,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(self.num_layers):
            x = getattr(self, f"conv_{i}")(x)
        return x
    

def get_sine_pos_encoding_1d(L, hidden_dim=128):
    # L: length of the sequence
    # hidden_dim: the dimensionality of the hidden representation
    device = L.device
    # use L as the indices
    indices = L.float()

    # compute the frequency for each element of the encoding
    freqs = torch.arange(0, hidden_dim // 2).float().to(device)
    freqs = torch.pow(10000, -2 * freqs / hidden_dim)

    # apply the sine function to the indices and frequencies
    pos_enc = torch.zeros(len(L), hidden_dim).to(device)
    pos_enc[:, 0::2] = torch.sin(torch.outer(indices, freqs))
    pos_enc[:, 1::2] = torch.cos(torch.outer(indices, freqs))

    return pos_enc

def feature_norm(desc2d, desc3d):
    '''
    norm features
    '''
    desc2d = desc2d.unsqueeze(0).permute(0, 2, 1)
    desc3d = desc3d.unsqueeze(0).permute(0, 2, 1)
    desc2d = nn.functional.normalize(desc2d, p=2, dim=1)
    desc3d = nn.functional.normalize(desc3d, p=2, dim=1)
    desc2d = desc2d.permute(0, 2, 1).squeeze()
    desc3d = desc3d.permute(0, 2, 1).squeeze()
    return desc2d, desc3d

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.
        Args:
            emb_indices: torch.Tensor (*)
        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings
