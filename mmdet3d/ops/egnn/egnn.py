import torch
from torch import nn as nn


class EGNNLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.edge_mlp  # l-nl-l-nl
        self.coord_mlp  # l-nl-l
        self.node_mlp  # l-nl-l-res

    def forward(self, x, h):
        B, N, D = h.shape
        x_neighbor = x.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, 3
        x_center = x.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, N, D
        h_neighbor = h.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, D
        h_center = h.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, N, D

        x_diff = x_center - x_neighbor  # B, N, N, 3
        x_dist2 = (x_diff * x_diff).sum(dim=-1, keepdim=True)  # B, N, N, 1

        edge_feat = torch.cat([
            h_center,
            h_neighbor,
            x_dist2,
        ], dim=-1)  # B, N, N, D+D+1+?
        m_edge = self.edge_mlp(edge_feat)  # B, N, N, D1

        x_res_weight = self.coord_mlp(m_edge)  # B, N, N, 3

        x_res = x_res_weight * (x_center - x_neighbor)  # B, N, N, 3
        x_update = x + x_res.sum(dim=2)  # B, N, 3

        neighbor = None
        m_edge_gathered = m_edge.gather(dim=2, index=neighbor)
        m_node = m_edge_gathered.sum(dim=2)  # B, N, D1

        node_feat = torch.cat([h_center, m_node], dim=-1)  # B, N, D+D1
        h_update = self.node_mlp(node_feat)  # B, N, D2

        return x_update, h_update
