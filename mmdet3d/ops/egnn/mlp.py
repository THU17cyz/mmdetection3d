from mmcv.cnn import ConvModule
from torch import nn as nn


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
                        norm_cfg=None,
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
