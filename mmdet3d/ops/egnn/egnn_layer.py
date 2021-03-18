import torch
from torch import nn as nn

# from mmdet3d.ops.knn import knn
from .attention_block import PointAttentionBlock
from .mlp import MLP


class EGNNLayer(nn.Module):

    def __init__(self,
                 edge_mlp=None,
                 coord_mlp=None,
                 node_mlp=None,
                 gather_mode=None,
                 neighbor_mode=None,
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
        self.neighbor_mode = neighbor_mode
        if self.neighbor_mode == 'knn':
            assert self.k is not None

    def forward(self,
                x_query,
                x_ref,
                h_query,
                h_ref,
                neighbor=None,
                mask=None):
        batch_size, xyz_dim, n_query = x_query.shape
        assert xyz_dim == 3
        feat_dim, n_ref = h_ref.shape[1:]

        if self.neighbor_mode == 'knn':
            raise NotImplementedError
            # neighbor = knn(self.k, x_query, x_ref, transposed=False)
            # neighbor = neighbor.transpose(3, 2).unsqueeze(1)
        elif self.neighbor_mode == 'ball':
            raise NotImplementedError
        elif self.neighbor_mode == 'all':
            pass
        else:
            assert neighbor is not None

        if self.neighbor_mode != 'all':
            x_ref = x_ref.unsqueeze(2).repeat(
                1, 1, n_query, 1)  # batch_size, 3, n_query, n_ref
            h_ref = h_ref.unsqueeze(2).repeat(
                1, 1, n_query, 1)  # batch_size, feat_dim, n_query, n_ref
            x_ref = x_ref.gather(
                dim=3, index=neighbor.repeat(1, xyz_dim, 1, 1))
            h_ref = h_ref.gather(
                dim=3, index=neighbor.repeat(1, feat_dim, 1, 1))
            n_ref = neighbor.shape[2]

        x_query = x_query.unsqueeze(3).repeat(
            1, 1, 1, n_ref)  # batch_size, 3, n_query, n_ref
        h_query = h_query.unsqueeze(3).repeat(
            1, 1, 1, n_ref)  # batch_size, feat_dim, n_query, n_ref

        x_diff = x_query - x_ref  # batch_size, 3, n_query, n_ref
        x_dist2 = (x_diff * x_diff).sum(
            dim=1, keepdim=True)  # batch_size, 1, n_query, n_ref

        edge_feat = torch.cat(
            [h_query, h_ref, x_dist2],
            dim=1)  # batch_size, feat_dim*2+1, n_query, n_ref
        m_edge = self.edge_mlp(
            edge_feat.reshape(batch_size, -1, n_query *
                              n_ref))  # batch_size, m_edge_dim, n_query, n_ref

        x_res_weight = self.coord_mlp(m_edge)
        x_res_weight = x_res_weight.reshape(
            batch_size, -1, n_query, n_ref)  # batch_size, 1, n_query, n_ref

        m_edge = m_edge.reshape(batch_size, -1, n_query, n_ref)
        #  m_edge_dim = m_edge.shape[1]

        x_res = x_res_weight * x_diff  # batch_size, 3, n_query, n_ref
        x_update = x_query[..., 0] + x_res.sum(dim=3)  # batch_size, 3, n_query

        if self.gather_mode == 'sum':
            if mask is None:
                m_node = m_edge.sum(dim=3)
            else:
                m_node = (m_edge * mask.unsqueeze(1)).sum(dim=3)
        elif self.gather_mode == 'max':
            m_node = m_edge.max(dim=3)
        elif self.gather_mode == 'avg':
            if mask is None:
                m_node = m_edge.avg(dim=3)  # batch_size, m_edge_dim, n_query
            else:
                m_node = (m_edge * mask.unsqueeze(1)).sum(
                    dim=3) / mask.unsqueeze(1).sum(dim=3)
        elif self.gather_mode == 'attention':
            rel_pos = x_diff
            rel_pos = rel_pos.reshape(batch_size, xyz_dim, -1)
            # TODO here we can also use m_edge as value
            attn = self.attention_block(h_query, h_ref, rel_pos, mask=mask)
            m_node = (m_edge * attn).sum(
                dim=3)  # batch_size, m_edge_dim, n_query
        else:
            raise NotImplementedError

        node_feat = torch.cat(
            [h_query, m_node],
            dim=1)  # batch_size, m_edge_dim + feat_dim, n_query
        h_update = self.node_mlp(node_feat)

        return x_update, h_update
