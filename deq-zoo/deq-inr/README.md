# $(\text{Implicit})^2$: Implicit Layers for Implicit Representations

This repository contains the code for $(\text{Implicit})^2$ (DEQ-INR) proposed in the paper [$(\text{Implicit})^2$: Implicit Layers for Implicit Representations](https://openreview.net/forum?id=AcoMwAU5c0s). 

## Requirements

Install dependencies via the following command.
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Download the data set from this [link](https://www.image-net.org/download.php) dataset.

## Training

To train DEQ-INR for video reconstruction, run the command.
```bash
python scripts/train_video.py --config configs/video/DEQ-siren_like-1L2048D.yaml
```

For other experiments (image, audio), run
```bash
python scripts/train_[TASK].py --config_file configs/[TASK]/[CONFIG_NAME].yaml
```

## Reference

```bib
@inproceedings{huang2021impsq,
  author    = {Zhichun Huang and Shaojie Bai and J. Zico Kolter},
  title     = {(Implicit)^2: Implicit Layers for Implicit Representations},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021},
}

@inproceedings{geng2021training,
    title = {On Training Implicit Models},
    author = {Zhengyang Geng and Xin-Yu Zhang and Shaojie Bai and Yisen Wang and Zhouchen Lin},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2021}
}
```