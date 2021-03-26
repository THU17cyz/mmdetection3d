import torch

from mmdet3d.ops import EGNNLayer


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
    print(x1, x2)
    assert torch.allclose(x2, x1, 1e-3)
