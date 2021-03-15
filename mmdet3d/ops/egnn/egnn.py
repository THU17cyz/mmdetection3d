import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.ops.knn import knn
from .attention_block import PointAttentionBlock


class MLP(nn.Module):

    def __init__(self,
                 in_channel=18,
                 conv_channels=(256, 256),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 last_act=True,
                 residual=False):
        super().__init__()
        self.mlp = nn.Sequential()
        self.residual = residual
        if residual:
            assert in_channel == conv_channels[-1], 'dimension mismatch'
        prev_channels = in_channel
        for i, conv_channel in enumerate(conv_channels):
            if not last_act and i == len(conv_channels) - 1:
                self.mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        prev_channels,
                        conv_channels[i],
                        1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                        bias=True))
            else:
                self.mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        prev_channels,
                        conv_channels[i],
                        1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        bias=True,
                        inplace=True))
            prev_channels = conv_channels[i]

    def forward(self, feats):
        if self.residual:
            return self.mlp(feats) + feats
        else:
            return self.mlp(feats)


class EGNNLayer(nn.Module):

    def __init__(self,
                 edge_mlp=None,
                 coord_mlp=None,
                 node_mlp=None,
                 gather_mode=None,
                 attention_block=None,
                 k=None):
        super().__init__()
        self.edge_mlp = MLP(**edge_mlp)  # l-nl-l-nl
        self.coord_mlp = MLP(**coord_mlp)  # l-nl-l
        self.node_mlp = MLP(**node_mlp)  # l-nl-l-res
        self.gather_mode = gather_mode
        self.k = k
        if gather_mode == 'attention':
            self.attention_block = PointAttentionBlock(**attention_block)

    def forward(self, x, h):
        B, N, D = h.shape
        x_neighbor = x.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, 3
        x_center = x.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, N, D
        h_neighbor = h.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, D
        h_center = h.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, N, D

        x_diff = x_center - x_neighbor  # B, N, N, 3
        x_dist2 = (x_diff * x_diff).sum(dim=-1, keepdim=True)  # B, N, N, 1

        edge_feat = torch.cat([h_center, h_neighbor, x_dist2],
                              dim=-1)  # B, N, N, D+D+1
        m_edge = self.edge_mlp(edge_feat)  # B, N, N, D1

        x_res_weight = self.coord_mlp(m_edge)  # B, N, N, 3

        x_res = x_res_weight * (x_center - x_neighbor)  # B, N, N, 3
        x_update = x + x_res.sum(dim=2)  # B, N, 3

        if self.gather_mode == 'knn':
            x_t = x.transpose(2, 1)
            neighbor = knn(self.k, x_t, x_t, transposed=True)
            neighbor = neighbor.transpose(2, 1).unsqueeze(-1).repeat(
                1, 1, 1, m_edge.shape[-1])
            m_edge_gathered = m_edge.gather(dim=2, index=neighbor)
            m_node = m_edge_gathered.sum(dim=2)  # B, N, D1
        elif self.gather_mode == 'attention':
            attn = self.attention_block(h, x)
            m_node = (m_edge * attn).sum(dim=2)
        else:
            raise NotImplementedError

        node_feat = torch.cat([h_center, m_node], dim=-1)  # B, N, D+D1
        h_update = self.node_mlp(node_feat)  # B, N, D2

        return x_update, h_update
