# Multiscale Deep Equilibrium Models

This repository contains the code for Multiscale Deep Equilibrium Models(MDEQ) proposed in the paper [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656). 

## Requirements

Install dependencies via the following command.
```bash
pip install -r requirements.txt
```

In addition, install the NVIDIA DALI library for faster ImageNet training.
```bash
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali==0.13.0
```

## Dataset Preparation (for ImageNet)

1. Download the [ImageNet](https://www.image-net.org/download.php) dataset.
2. Follow the [instruction](https://mxnet.apache.org/versions/1.0.0/tutorials/vision/large_scale_classification.html) to produce the recordIO format dataset for faster loading (strongly ecommended, ~1ms data latency w/ DALI loader).

## Training

### CIFAR10

To train MDEQ on the CIFAR10 dataset, run the command under `mdeq_cifar`.
```bash
bash run.sh
```

### ImageNet

To train MDEQ on the ImageNet dataset using Pytorch DDP, run the command under `mdeq_imagenet`. 
```bash
bash run_mdeq.sh --data YOUR_DATA_PATH
```
The training script requires 2 GPUs. More GPUs can lead to faster training. You can range the number of GPUs by setting up `--nproc_per_node`. Batch size per GPU needs to be adjusted accordingly using `--batch-size`.

## Evaluation

Download the pretrained checkpoints from this [link](https://drive.google.com/drive/folders/1vv9dkxO-wX4e47u8aQjC62wiYhEKSajA?usp=sharing). Copy the `FILE_ID` and run this command to fetch a checkpoint to your local machine.

```bash
wget https://drive.google.com/uc?id=FILE_ID
```

### CIFAR10

To evaluate the MDEQ checkpoint on the CIFAR10 dataset, run the command under `mdeq_cifar`.
```bash
bash run.sh --eval --test_model PATH_TO_YOUR_CKPT
```

You could have the top-1 accuracy of 95.0 on CIFAR10.

### ImageNet

To evaluate MDEQ on the ImageNet dataset using Pytorch DDP, run the command under `mdeq_imagenet`. 
```bash
bash run_mdeq.sh --eval --data YOUR_DATA_PATH --resume YOUR_CKPT_PATH
```
The evaluation script will consume 2 GPUs. More GPUs can lead to faster evaluation. You can range the number of GPUs by setting up `--nproc_per_node`.

You could have the top-1 accuracy of 75.7 on ImageNet.

To accelerate the inference, you can range max solver steps/number of equilibrium function calls (NFEs) by setting up `--eval_f_max_iter`. 

Taking `--eval_f_max_iter 20` will achieve the top-1 accuracy of 75.7 and fatser inference speed.

Taking `--eval_f_max_iter 10` will achieve the top-1 accuracy of 73.6 and much faster inference speed.

## Reference

```bib
@inproceedings{bai2020multiscale,
  author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
  title     = {Multiscale Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
}

@inproceedings{geng2021training,
  title = {On Training Implicit Models},
  author = {Zhengyang Geng and Xin-Yu Zhang and Shaojie Bai and Yisen Wang and Zhouchen Lin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
```