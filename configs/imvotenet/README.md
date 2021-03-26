# ImVoteNet: Boosting 3D Object Detection in Point Clouds with Image Votes

## Introduction

[ALGORITHM]

We implement ImVoteNet and provide the result and checkpoints on SUNRGBD.

```
@inproceedings{qi2020imvotenet,
  title={Imvotenet: Boosting 3d object detection in point clouds with image votes},
  author={Qi, Charles R and Chen, Xinlei and Litany, Or and Guibas, Leonidas J},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={4404--4413},
  year={2020}
}
```

## Results

### SUNRGBD (Stage 1, image network pre-train)

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_stage2_16x8_sunrgbd-3d-10class.py)     |  3x    |8.1||59.07||[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238.log.json)|

### SUNRGBD (Stage 2)

|  Backbone   | Lr schd | Mem (GB) | Inf time (fps) | AP@0.25 |AP@0.5| Download |
| :---------: | :-----: | :------: | :------------: | :----: |:----: | :------: |
|    [PointNet++](./imvotenet_stage2_16x8_sunrgbd-3d-10class.py)     |  3x    |8.1||59.07||[model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0.pth) &#124; [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/votenet/votenet_16x8_sunrgbd-3d-10class/votenet_16x8_sunrgbd-3d-10class_20200620_230238.log.json)|
