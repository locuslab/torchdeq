# Deep Equilibrium Optical Flow Estimation

This repository contains the code for DEQ-Flow proposed in the paper [Deep Equilibrium Optical Flow Estimation](https://openreview.net/forum?id=AcoMwAU5c0s). 

## Requirements

Install dependencies via the following command.

```bash
conda create --name deq python==3.7.9
conda activate deq
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install tensorboard scipy opencv matplotlib einops -c conda-forge
```

## Dataset Preparation

Download the following datasets into the `data` directory.

- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
- [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
- [MPI Sintel](http://sintel.is.tue.mpg.de/)
- [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
- [HD1k](http://hci-benchmark.iwr.uni-heidelberg.de/)

Organize your datasets as follows. 

```Shell
├── data
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
```

## Pipeline Explained

Optical flow is the motion of objects between two consecutive frames. Having an image pair or a sequence of images of tensor shape (H, W), optical flow is presented as a vector field of tensor shape (H, W, 2) evolution along the time dimension, standing for the change of each pixel in the first frame to its corresponding position in the second frame.

Given the difficulties of obtaining ground truth optical flow for arbitrary image pairs, modern optical flow models are usually trained on synthetic datasets and evaluated on real datasets of sparse annotations collected by laser.

The standard training pipeline for optical flow estimation is first to pretrain the model using the [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs) dataset and then train on [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) for large motions. Both training datasets are synthetic. The pretrained optical flow model will be evaluated on the high-quality animated dataset [MPI Sintel](http://sintel.is.tue.mpg.de/) and the real-world driving dataset [KITTI](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow).

## Training

The following commands produce the training pipeline explained above.

To train DEQ-Flow-B, run this command.

```bash
bash train_B.sh
```

The training requires 2 GPUs. 

To train DEQ-Flow-H, run this command.

```bash
bash train_H.sh
```

The training requires 3 GPUs. 

## Evaluation

To evaluate a pretrained model, add the `--eval` flag. A checkpoint can be loaded using the `--restore_ckpt` flag, followed by the path to the checkpoint, `--restore_ckpt CKPT_PATH`.

Configure the datasets you want to use for evaluation using the `--validation` flag. The common datasets include `kitti` and `sintel`.

### Pretrained Checkpoint

You can download the checkpoints from this [link](https://drive.google.com/drive/folders/1RTTKy9iQm4pYCEi2Zqzbxi9MHmASWeh8?usp=sharing). Run the following command to evaluate pretrained checkpoints on Sintel and KITTI datasets. This is a reference [log](https://github.com/locuslab/torchdeq/blob/main/deq-zoo/deq-flow/log_dir/val.txt).

```bash
bash val.sh
```

DEQ-Flow-B has the same model size as [RAFT](https://github.com/princeton-vl/RAFT/tree/master). DEQ-Flow-H is a model of twice the width compared to DEQ-Flow-B regarding the DEQ layer. We've trained DEQ-Flow-H for two schedules, 120k on Chairs + 120k on Things (1x) and 120k on Chairs + 360k on Things (3x). As a baseline, we also benchmark RAFT-H (the same model size as DEQ-Flow-H) using the 1x schedule.

|  Checkpoint Name | Sintel (clean) | Sintel (final) | KITTI AEPE  | KITTI F1-all |
| :--------------- | :------------: | :------------: | :---------: | :----------: |
| DEQ-Flow-B    | 1.39 | 2.77 | 4.46 | 14.18 |
| RAFT-H-1x     | 1.36 | 2.59 | 4.47 | 16.16 |
| DEQ-Flow-H-1x | 1.27 | 2.59 | 3.76 | 12.97 |
| DEQ-Flow-H-3x | 1.27 | 2.48 | 3.77 | 13.41 |

DEQ-Flow demonstrates a clear performance and efficiency margin and much stronger scaling property (scale up to larger models) over RAFT!

## Reference

```bib
@inproceedings{deq-flow,
    author = {Bai, Shaojie and Geng, Zhengyang and Savani, Yash and Kolter, J. Zico},
    title = {Deep Equilibrium Optical Flow Estimation},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}

@inproceedings{geng2021training,
    title = {On Training Implicit Models},
    author = {Zhengyang Geng and Xin-Yu Zhang and Shaojie Bai and Yisen Wang and Zhouchen Lin},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2021}
}
```