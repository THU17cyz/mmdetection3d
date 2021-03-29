import torch

from mmdet3d.ops import EGNNLayer, PointEGNNModule


def test_layer_equivariance():
    edge_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))

    coord_mlp = dict(
        last_act=False,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))

    node_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))

    mlp_dims = [[[65, 32, 32], [32, 32, 32], [64, 32, 32]],
                [[257, 128, 128], [128, 128, 32], [256, 128, 128]],
                [[257, 128, 128], [128, 128, 32], [256, 128, 128]],
                [[257, 128, 128], [128, 128, 32], [256, 128, 256]]]

    egnn_layer = EGNNLayer(
        edge_mlp=edge_mlp,
        coord_mlp=coord_mlp,
        node_mlp=node_mlp,
        mlp_dims=mlp_dims[0],
        neighbor_mode='all',
        gather_mode='max',
    ).cuda()

    x_ref = torch.randn((2, 3, 16, 8)).cuda()
    x_query = x_ref[..., 0]

    h_ref = torch.randn((2, 32, 16, 8)).cuda()
    h_query = h_ref[..., 0]

    x1, h1 = egnn_layer(x_query, x_ref, h_query, h_ref)

    trans1 = torch.randn((3, 1, 1)).cuda()
    trans2 = torch.randn((3, 1, 1)).cuda()
    x_ref += trans1
    mat = torch.tensor([[1., 0., 0.], [0., 0.5, 0.86602540378],
                        [0., -0.86602540378, 0.5]]).cuda()

    x_ref = x_ref.view(2, 3, -1).transpose(1, 2).contiguous() @ mat
    x_ref = x_ref.transpose(1, 2).contiguous().view(2, 3, 16, -1)
    x_ref += trans2
    x_query = x_ref[..., 0]

    x2, h2 = egnn_layer(x_query, x_ref, h_query, h_ref)

    x1 += trans1[..., 0]
    x1 = x1.transpose(1, 2).contiguous() @ mat
    x1 = x1.transpose(1, 2).contiguous()
    x1 += trans2[..., 0]

    assert torch.allclose(h2, h1, 1e-3)
    # print(x1, x2)
    assert torch.allclose(x2, x1, 1e-3)


def test_module_equivariance1():
    mlp_dims = [[65, 32, 32], [32, 32, 32], [64, 32, 32]]
    edge_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))

    coord_mlp = dict(
        last_act=False,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))

    node_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='GN', num_groups=2))
    egnn_module = PointEGNNModule(
        num_point=32,
        radius=0.2,
        num_sample=16,
        egnn_layer_cfg=dict(
            mlp_dims=mlp_dims,
            edge_mlp=edge_mlp,
            coord_mlp=coord_mlp,
            node_mlp=node_mlp,
            gather_mode='max',
            neighbor_mode='all')).cuda()

    xyz = torch.randn((2, 64, 3)).cuda()
    features = torch.randn((2, 32, 64)).cuda()
    xyz_new, xyz_shifted, features_new, indices = \
        egnn_module(xyz, features)

    trans1 = torch.randn((3, )).cuda()
    trans2 = torch.randn((3, )).cuda()
    xyz += trans1
    mat = torch.tensor([[1., 0., 0.], [0., 0.5, 0.86602540378],
                        [0., -0.86602540378, 0.5]]).cuda()

    xyz = xyz.view(-1, 3) @ mat
    xyz = xyz.view(2, -1, 3)
    xyz += trans2

    xyz_new_2, xyz_shifted_2, features_new_2, indices_2 = \
        egnn_module(xyz, features)

    xyz_new += trans1
    xyz_new = xyz_new.view(-1, 3) @ mat
    xyz_new = xyz_new.view(2, -1, 3)
    xyz_new += trans2

    xyz_shifted = xyz_shifted.transpose(1, 2).contiguous()
    xyz_shifted += trans1
    xyz_shifted = xyz_shifted.view(-1, 3) @ mat
    xyz_shifted = xyz_shifted.view(2, -1, 3)
    xyz_shifted += trans2
    xyz_shifted = xyz_shifted.transpose(1, 2).contiguous()

    assert torch.allclose(features_new, features_new_2, 1e-3)
    assert torch.allclose(indices.float(), indices_2.float(), 1e-3)
    assert torch.allclose(xyz_new, xyz_new_2, 1e-3)
    assert torch.allclose(xyz_shifted, xyz_shifted_2, 1e-3)


def test_module_equivariance2():
    mlp_dims = [[65, 32, 32], [32, 32, 32], [64, 32, 32]]
    edge_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='BN1d'))

    coord_mlp = dict(
        last_act=False,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='BN1d'))

    node_mlp = dict(
        last_act=True,
        residual=False,
        act_cfg=dict(type='Swish'),
        conv_cfg=dict(type='Conv1d'),
        # norm_cfg=dict(type='BN1d'))
        norm_cfg=dict(type='BN1d'))
    egnn_module = PointEGNNModule(
        num_point=32,
        radius=0.2,
        num_sample=16,
        egnn_layer_cfg=dict(
            mlp_dims=mlp_dims,
            edge_mlp=edge_mlp,
            coord_mlp=coord_mlp,
            node_mlp=node_mlp,
            gather_mode='max',
            neighbor_mode='all')).cuda()

    xyz = torch.randn((2, 64, 3)).cuda()
    features = torch.randn((2, 32, 64)).cuda()
    xyz_new, xyz_shifted, features_new, indices = \
        egnn_module(xyz, features)

    trans1 = torch.randn((3, )).cuda()
    trans2 = torch.randn((3, )).cuda()
    xyz += trans1
    mat = torch.tensor([[1., 0., 0.], [0., 0.5, 0.86602540378],
                        [0., -0.86602540378, 0.5]]).cuda()

    xyz = xyz.view(-1, 3) @ mat
    xyz = xyz.view(2, -1, 3)
    xyz += trans2

    xyz_new_2, xyz_shifted_2, features_new_2, indices_2 = \
        egnn_module(xyz, features)

    xyz_new += trans1
    xyz_new = xyz_new.view(-1, 3) @ mat
    xyz_new = xyz_new.view(2, -1, 3)
    xyz_new += trans2

    xyz_shifted = xyz_shifted.transpose(1, 2).contiguous()
    xyz_shifted += trans1
    xyz_shifted = xyz_shifted.view(-1, 3) @ mat
    xyz_shifted = xyz_shifted.view(2, -1, 3)
    xyz_shifted += trans2
    xyz_shifted = xyz_shifted.transpose(1, 2).contiguous()

    assert torch.allclose(features_new, features_new_2, 1e-3)
    assert torch.allclose(indices.float(), indices_2.float(), 1e-3)
    assert torch.allclose(xyz_new, xyz_new_2, 1e-3)
    assert torch.allclose(xyz_shifted, xyz_shifted_2, 1e-3)
