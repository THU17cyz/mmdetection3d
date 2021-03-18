import torch
# from mmcv.cnn import ConvModule
from torch import nn as nn
# from torch.nn import functional as F
from typing import List

from ..group_points import (KNNAndGroup, Points_Sampler, QueryAndGroup,
                            gather_points)
from .egnn_layer import EGNNLayer


class PointEGNNModuleMSG(nn.Module):
    """Point set abstraction module with multi-scale grouping used in
    Pointnets.

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point: int,
                 ks: List[int] = None,
                 radii: List[float] = None,
                 sample_nums: List[int] = None,
                 egnn_layer_cfgs: List[dict] = None,
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False):
        super().__init__()
        if radii is not None:
            assert len(radii) == len(sample_nums) == len(egnn_layer_cfgs)
        else:
            assert ks is not None
            assert len(ks) == len(egnn_layer_cfgs)
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(egnn_layer_cfgs, dict):
            egnn_layer_cfgs = [egnn_layer_cfgs]

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.groupers = nn.ModuleList()
        self.egnn_layers = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        self.points_sampler = Points_Sampler(self.num_point, self.fps_mod_list,
                                             self.fps_sample_range_list)

        if radii is not None:
            for i in range(len(radii)):
                radius = radii[i]
                sample_num = sample_nums[i]
                if num_point is not None:
                    if dilated_group and i != 0:
                        min_radius = radii[i - 1]
                    else:
                        min_radius = 0
                    grouper = QueryAndGroup(
                        radius,
                        sample_num,
                        min_radius=min_radius,
                        use_xyz=False,
                        subtract_center=False,
                        return_grouped_xyz=True,
                        return_unique_cnt=True)
                else:
                    # TODO
                    raise NotImplementedError
                self.groupers.append(grouper)
            self.group_mode = 'ball'
        else:
            for i in range(len(ks)):
                k = ks[i]
                if num_point is not None:
                    grouper = KNNAndGroup(
                        k,
                        use_xyz=False,
                        subtract_center=False,
                        return_grouped_xyz=True)
                else:
                    # TODO
                    raise NotImplementedError
                self.groupers.append(grouper)
            self.group_mode = 'knn'

        for i in range(len(egnn_layer_cfgs)):
            egnn_layer_cfg = egnn_layer_cfgs[i]
            egnn_layer = EGNNLayer(**egnn_layer_cfg)
            self.egnn_layers.append(egnn_layer)

    def forward(
        self,
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        indices: torch.Tensor = None,
        target_xyz: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []
        new_xyz_shifted_list = []
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            dsamp_xyz = gather_points(
                xyz_flipped, indices) if self.num_point is not None else None
            dsamp_xyz_t = dsamp_xyz.transpose(1, 2).contiguous()
        elif target_xyz is not None:
            dsamp_xyz = target_xyz.contiguous()
        else:
            indices = self.points_sampler(points_xyz, features)
            dsamp_xyz = gather_points(
                xyz_flipped, indices) if self.num_point is not None else None
            dsamp_xyz_t = dsamp_xyz.transpose(1, 2).contiguous()

        dsamp_features = gather_points(features, indices)

        for i in range(len(self.groupers)):
            # (B, C, num_point, nsample)
            if self.group_mode == 'knn':
                new_features, new_xyz = self.groupers[i](points_xyz,
                                                         dsamp_xyz_t, features)
                mask = None
            else:
                new_features, new_xyz, mask = self.groupers[i](points_xyz,
                                                               dsamp_xyz_t,
                                                               features)

            # (B, mlp[-1], num_point)
            new_xyz_shifted, new_features = self.egnn_layers[i](
                dsamp_xyz, new_xyz, dsamp_features, new_features, mask=mask)

            new_features_list.append(new_features)

            new_xyz_shifted_list.append(new_xyz_shifted)

        return dsamp_xyz, torch.cat(
            new_xyz_shifted_list, dim=1), torch.cat(
                new_features_list, dim=1), indices


class PointEGNNModule(PointEGNNModuleMSG):
    """Point set abstraction module used in Pointnets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 egnn_layer_cfg: dict = None,
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super().__init__(
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            egnn_layer_cfgs=[egnn_layer_cfg],
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list)
