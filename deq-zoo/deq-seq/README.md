# Deep Equilibrium (DEQ) Sequence Models

This repository contains the code of the deep equilibrium model (DEQ) for Language Modeling proposed in the paper [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377).

## Requirements

PyTorch >=1.11.0 recommended

### Dataset Preparation

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

To train the model using implicit differentiation (IFT), run the following command:
```sh
bash deq_seq_ift.sh --name ift
```
You should expect to get a test perplexity around 23.8 with this setting.

To train the model using implicit differentiation and Jacobian regularization, run the following command:
```sh
bash deq_seq_ift_jr.sh --name ift_jr
```

In practice, we can do this efficiently by regularizing `||J_f||_F` (which characterizes fixed point models' stability) using the Hutchinson estimator. In practice, we can apply this regularization stochastically and adjust its strength dynamically. Please refer to [Stabilizing Equilibrium Models by Jacobian Regularization](https://arxiv.org/abs/2106.14342) for more details.

When training the model, the Jacobian regularization settings can be controlled entirely by the `argparse` options, see options using `--help`.

## Evaluation

To evaluate a pre-trained model, add the `--eval` flag.

```bash
bash deq_seq_ift.sh --eval --name [NAME_YOUR_RUN] --load_path [CHECKPOINT_NAME].pth
```

At training time, the default mem_len is 150, while you can set a larger memory window length for better test time performance using the `--mem_len` flag. Typically, a memory length of 480 will be used for evaluating language modeling performance on the Wikitext-103 dataset.

### Pretrained Checkpoints

You can download the checkpoints from this [link](https://drive.google.com/drive/folders/1GmrgRu6Hr2ZczbenaBR6-wH_k4-tNCKA?usp=sharing). Here are the evaluation statistics from different `mem_len`.

| Model | Parameters | `mem_len` | Perplexity |
| :---: | :--------: | :-------: | :--------: |
| model-ift.pth    | 98M | 150 | 23.78 |
| model-ift.pth    | 98M | 480 | 23.31 |
| model-ift-jr.pth | 98M | 150 | 23.67 |
| model-ift-jr.pth | 98M | 480 | 23.16 |

`model-ift.pth` was trained using IFT. Please refer to `deq_seq_ift.sh` for the setup.

`model-ift-jr.pth` was trained using IFT w/ Jacobian Regularization. Please refer to `deq_seq_ift_jr.sh` for the setup.

## Reference

```bib
@inproceedings{bai2019deep,
  author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title     = {Deep Equilibrium Models},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2019},
}

@inproceedings{bai2021stabilizing,
  title     = {Stabilizing Equilibrium Models by Jacobian Regularization},
  author    = {Shaojie Bai and Vladlen Koltun and J. Zico Kolter},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2021}
}
```