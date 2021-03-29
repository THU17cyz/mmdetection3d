import torch
from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import PointCoordsFPModule, PointEGNNModule, PointFPModule
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet


@BACKBONES.register_module()
class EGNNSASSG(BasePointNet):
    """PointNet2 with Single-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(64, 32, 16, 16),
                 egnn_layer_cfgs=None,
                 fp_channels=((256, 256), (256, 256)),
                 fp_norm_cfg=dict(type='BN2d')):

        super().__init__()
        self.num_sa = len(egnn_layer_cfgs)
        self.num_fp = len(fp_channels)

        # sa_channels = ((64, 64, 128), (128, 128, 256), (128, 128, 256),
        #                (128, 128, 256))

        # assert len(num_points) == len(radius) == len(num_samples) == len(
        #     sa_channels)
        assert self.num_sa >= self.num_fp

        self.SA_modules = nn.ModuleList()
        # sa_in_channel = in_channels - 3  # number of channels without xyz

        sa_in_channel = in_channels
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            # cur_sa_mlps = list(sa_channels[sa_index])
            # cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = egnn_layer_cfgs[sa_index].mlp_dims[-1][-1]
            self.SA_modules.append(
                PointEGNNModule(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    egnn_layer_cfg=egnn_layer_cfgs[sa_index],
                ))

            # cur_sa_mlps = list(sa_channels[sa_index])
            # cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            # sa_out_channel = cur_sa_mlps[-1]

            # self.SA_modules.append(
            #     build_sa_module(
            #         num_point=num_points[sa_index],
            #         radius=radius[sa_index],
            #         num_sample=num_samples[sa_index],
            #         mlp_channels=cur_sa_mlps,
            #         norm_cfg=dict(type='BN2d'),
            #         cfg=dict(
            #          type='PointSAModule',
            #          pool_mod='max',
            #          use_xyz=True,
            #          normalize_xyz=True)))

            skip_channel_list.append(sa_out_channel)
            sa_in_channel = sa_out_channel

        self.FP_modules = nn.ModuleList()
        self.CoordsFP_Module = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(
                PointFPModule(mlp_channels=cur_fp_mlps, norm_cfg=fp_norm_cfg))
            self.CoordsFP_Module.append(PointCoordsFPModule())
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of \
                    each fp features.
                - fp_features (list[torch.Tensor]): The features \
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]

        features = xyz.new_ones((batch, 2, num_points))

        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_xyz_shifted = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_xyz_shifted, cur_features, cur_indices = \
                self.SA_modules[i](sa_xyz[i], sa_features[i],
                                   xyz_to_update=sa_xyz_shifted[i])
            sa_xyz.append(cur_xyz)
            sa_xyz_shifted.append(cur_xyz_shifted.transpose(1, 2).contiguous())
            sa_features.append(cur_features)
            # print(i, cur_features.min())
            # print(i, cur_features.max())
            # print(i, cur_xyz_shifted.transpose(1, 2).contiguous().min())
            # print(i, cur_xyz_shifted.transpose(1, 2).contiguous().max())
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        fp_xyz = [sa_xyz[-1]]
        fp_xyz_shifted = [sa_xyz_shifted[-1].transpose(1, 2).contiguous()]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz_shifted.append(self.CoordsFP_Module[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                fp_xyz_shifted[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        # ret = dict(
        #     fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)

        ret = dict(
            sa_xyz=sa_xyz[1:],
            sa_xyz_shifted=sa_xyz_shifted[1:],
            # sa_features=sa_features,
            fp_xyz_shifted=fp_xyz_shifted,
            fp_xyz=fp_xyz,
            fp_features=fp_features,
            fp_indices=fp_indices)
        return ret

    def _split_point_feats(self, points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """
        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features
