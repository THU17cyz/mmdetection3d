import torch
from torch import nn

from .mlp import MLP

class PointAttentionBlock(nn.Module):

    def __init__(self,
                 to_q=None,
                 to_kv=None,
                 pos_mlp=None,
                 attn_mlp=None,
                 use_pos=True):
        super().__init__()

        self.to_q = MLP(**to_q)

        self.to_kv = MLP(**to_kv) # bias = False?why

        self.attn_mlp = MLP(**attn_mlp)

        self.use_pos = use_pos
        if self.use_pos:
            self.pos_mlp = MLP(**pos_mlp)

    def forward(self, h_query, h_ref, rel_pos=None, mask=mask):
        # get queries, keys, values
        q = self.to_q(h_query[..., 0])
        k, v = self.to_kv(h_ref[:, :, 0, :]).chunk(2, dim=1)

        # calculate relative positional embeddings
        if self.use_pos:
            assert rel_pos is not None
            rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is
        # a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, :, None] - k[:, :, None, :]

        # v = v[:, None, :, :]

        # # add relative positional embeddings to value
        # v = v + rel_pos_emb

        # use attention mlp, making sure to
        # add relative positional embedding first
        if self.use_pos:
            qk_rel = qk_rel + rel_pos_emb

        qk_rel = qk_rel.reshape(qk_rel.shape[0], -1, qk_rel.shape[-1])
        sim = self.attn_mlp(qk_rel).reshape(qk_rel.shape[0], -1, q.shape[2], k.shape[2])
        # attention
        if mask is not None:
            mask_value = torch.finfo(sim.dtype).min
            sim.masked_fill_(~mask.unsqueeze(1).repeat(1, sim.shape[1], 1, 1), mask_value)
        attn = sim.softmax(dim=3)

        return attn