# Models in DEQ Zoo

[`deq-zoo`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo) currently supports six implicit models via TorchDEQ. For each project, we provide a README doc for data preparation and launching instructions.

## DEQ

The first [Deep Equilibrium Model](https://arxiv.org/abs/1909.01377) is a sequence model that takes advantage of transformers in its model design. Given the injection $U(\mathbf{x}_{0:T})$ from the input sequence and the past context $\mathbf{z}^\star_{0:t}$, DEQ transformer predicts the next tokens via the fixed points $\mathbf{z}^\star_{t:T}$ of a transformer block,

$$
\begin{array}{llll}
& \mathbf{q}, \mathbf{k}, \mathbf{v} & = & \mathbf{w} \mathbf{z}^\star_{0:T} + U(\mathbf{x}_{0:T}) \\
& \tilde{\mathbf{z}}       & = & \mathbf{z}^\star_{t:T} + \text{Attention}\left(\mathbf{q}, \mathbf{k}, \mathbf{v}\right)   \\ 
& \mathbf{z}^\star_{t:T}     & = & \tilde{\mathbf{z}}   + \text{FFN}\left(\tilde{\mathbf{z}} \right)  \\
\end{array} 
$$

where Attention is MultiHead Decoder Attention, FFN is a 2-layer feed-forward network.

In DEQ Zoo, we implement the DEQ transformer and benchmark it through the word-level language modeling on WikiText-103~\cite{wiki}. The model details and training protocols are redesigned based on TorchDEQ.

- [`deq-seq`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-seq): Language modeling on WikiText-103. Implementation using Pytorch DataParallel.
- [`deq-lm`]((https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-seq)): Faster & updated implementation using PyTorch Distributed Data Parallel (DDP) framework. This is the recommended version.

## MDEQ

This directory contains the code for Multiscale Deep Equilibrium Models(MDEQ) proposed in the paper [Multiscale Deep Equilibrium Models](https://arxiv.org/abs/2006.08656). 

- [`mdeq`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/mdeq): Code for training MDEQs on CIFAR10 and ImageNet (DDP).

## IGNN

This directory contains the code for Implicit Graph Neural Networks (IGNN) proposed in the paper [Implicit Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf).

- [`ignn`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/ignn): Code for conducting graph and node classification tasks, using datasets like PPI.

## DEQ-Flow

[Deep Equilibrium Optical Flow Estimation](https://arxiv.org/abs/2204.08442)

- [`deq-flow`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-flow): Code for training and evaluating optical flow models.

## DEQ-INR

[$(\text{Implicit})^2$: Implicit Layers for Implicit Representations](https://openreview.net/forum?id=AcoMwAU5c0s).

- [`deq-inr`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/mdeq): Code for converting and compressing image, audio, and video data into implicit layers for implicit representations.

## DEQ-DDIM 

[Deep Equilibrium Approaches to Diffusion Models](https://arxiv.org/abs/2210.12867)

- [`deq-ddim`](https://github.com/locuslab/torchdeq/tree/main/deq-zoo/deq-ddim): Code for performing parallel diffusion sampling & inversion using the joint equilibrium of the sampling trajectory.
