# Deep Equilibrium (DEQ) Language Models

This repository contains TorchDEQ's implementation of the deep equilibrium model (DEQ) for Language Modeling proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377). 

Better model design. Stable training. Faster implementation using Pytorch DDP.

## Requirements

PyTorch>=1.11.0 recommended.

## Dataset Preparation

You can download the Wikitext-103 dataset using 
```bash
bash get_data.sh
```

## Model Explained

For sequence modeling in particular, the default arguments to pass into the DEQ only include `z_now` (the hidden sequence we drive to equilibrium), `u_cat` (input injection sequence), and `z_hist` (history padding to append to the left of `z_now`). Graphically:

```
  [<-------------- u_cat ------------->]
  [                  |                 ]         
  [<==== z_hist ====>|<==== z_now ====>]
  [    (mem_len=L)   |   (seq_len=L')  ]
(l=0)              (l=L)             (l=L+L')
```

## Training

To train the DEQ transformer, run the following command:

```bash
bash run.sh 4 [PORT_NUM] --name [NAME_YOUR_RUN]
```

The training script will consume 4 GPUs. You should expect 6700MB memory usage per GPU.

## Evaluation

To evaluate a pretrained model, use the `--eval` flag.

```bash
bash run.sh 4 [PORT_NUM] --eval --name [NAME_YOUR_RUN] --load_path [CHECKPOINT_NAME].pth
```

At training time, the default mem_len is 150, while you can set a larger memory window length for better test time performance using the `--mem_len` flag. Typically, a memory length of 480 will be used for evaluating language modeling performance on the Wikitext-103 dataset.

### Pretrained Checkpoint

You can download the checkpoints from this [link](https://drive.google.com/drive/folders/147fXBiSUmfNPCiXEsWxHtbwM5-y5YMlN?usp=sharing). Here are the evaluation statistics from different `mem_len`.

|  Model | Parameters | `mem_len` | Perplexity |
| :----: | :--------: | :-------: | :--------: |
| DEQ-LM | 98M | 150 | 22.34 |
| DEQ-LM | 98M | 480 | 21.85 |


## Reference

```bib
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}

@misc{torchdeq,
    author = {Zhengyang Geng and J. Zico Kolter},
    title = {TorchDEQ: A Library for Deep Equilibrium Models},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/locuslab/torchdeq}},
}
```