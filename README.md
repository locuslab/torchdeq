<img src="https://drive.google.com/uc?id=1tCXsv0yanvQqncB9ke8x6fCKonu4z4iB" alt="TorchDEQ Logo" width="65" align="left"><div align="center"><h1>TorchDEQ: A Library for Deep Equilibrium Models</h1></div>

<p align="center">
| <a href=""><b>Documentation</b></a> | <a href="https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing"><b>Colab Tutorial</b></a> | <a href=""><b>DEQ Zoo</b></a>  | <a href="TODO.md"><b>Roadmap</b></a> | <a href="README.md#citation"><b>Citation</b></a> |
</p>

## Introduction

Deep Equilibrium Models, or DEQs, a recently developed class of implicit neural networks, merge the concepts of fixed point systems with modern deep learning. Fundamentally, DEQ models establish their output based on the equilibrium of nonlinear systems. This can be represented as:

$$\mathbf{z}^\star=f_\theta(\mathbf{z}^\star, \mathbf{x})$$

Here, $\mathbf{x}$ is the input fed into the network, while $\mathbf{z}^\star$ stands as its output.

Enter **TorchDEQ** - a fully featured, out-of-the-box, and PyTorch-based library tailored for the design and deployment of DEQs. It provides intuitive, decoupled, and modular interfaces to customize general-purpose DEQs for arbitrary tasks, all with just a handful of code lines.

Dive into the world of DEQ with TorchDEQ! Craft your own DEQ effortlessly in just a single line of code. Kickstart your journey with our [Colab Tutorial](https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing) — best enjoyed with a comforting cup of tea!

## Installation

- Through pip.

    ```bash
    pip install torchdeq
    ```

- From source.

    ```bash
    git clone https://github.com/locuslab/torchdeq.git && cd torchdeq
    pip install -e .
    ```

## Quick Start

- Automatic arg parser decorator. You can call this function to add commonly used DEQ args to your program. 

```Python
add_deq_args(parser)
```

- Automatic DEQ instantiation. Call `get_deq` to get your DEQ layer in a single line! It's highly decoupled implementation agnostic to your model design.

```Python
deq = get_deq(args)
```

- Easy DEQ forward. Even for a multi-equilibria system, you can execute your DEQ forward in a single line!

```Python
# Assume f is a functioin of three tensors a, b, c.
def fn(a, b, c):
    # Do something here...
    # Having the same input and output tensor shapes.
    return a, b, c

# A callable object (`fn` here) that defines your fixed point system.
# `fn` can be a functor defined in your Pytorch forward function.
# A functor can take your input injection from the local variables. 
# You can also pass a Pytorch Module into the DEQ class.
z_out, info = deq(fn, (a0, b0, c0))
```

- Automatic DEQ backward. Gradients (both exact and inexact grad) are tracked automatically! Working with TorchDEQ is the same as other standard PyTorch operators. Just post-process ``z_out`` as normal tensors!

## Contributions

We warmly welcome contributions to TorchDEQ from the community! If you have suggestions for improving the library, introducing new features, or identifying and fixing bugs, please open an issue to discuss with us! Once a direction has been discussed, we can proceed to build, test, and submit a pull request (PR) to TorchDEQ together. Keep a PR clean, well-tested, and have a single focus! While numerical errors and stability may seem minor initially, they can culminate in significant effects over time.

We have provided a preliminary [roadmap](TODO.md) for the development of this library and are always open to fresh perspectives. Feel free to reach out for questions, discussions, or library developments! Here is my [email](zhengyanggeng@gmail.com).

## Logo Explained

The logo we’ve chosen draws inspiration from the ancient symbol, *[Ouroboros](https://en.wikipedia.org/wiki/Ouroboros)*, a powerful emblem depicting a serpent or dragon eternally consuming its own tail. Unearthed in the tomb of Tutankhamun, the Ouroboros symbolizes the cyclicality of time, embodying both creation and destruction, inception and conclusion. It’s a profound representation of infinity and wholeness, transcending various mythologies and philosophies across time.

For DEQ models, our choice of logo bears a metaphorical weight. The dragon, denoting $f(\mathbf{x})$, biting its tail, representing $\mathbf{x}$, paints a vivid picture of a function attaining a fixed point. It's a metaphor layered with meaning, visualizing the attainment of stability, illustrated by the dragon completing its circle by biting its tail. This symbol is not just a snapshot of equilibrium; it's a dynamic representation of the infinite nature inherent in DEQ models.

## Citation

```bibtex
@misc{torchdeq,
    author = {Zhengyang Geng and J. Zico Kolter},
    title = {TorchDEQ: A Library for Deep Equilibrium Models},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/locuslab/torchdeq}},
}
```

## Acknowledgements

This codebase is largely inspired by remarkable projects from the community.
We would like to sincerely thank [DEQ](https://github.com/locuslab/deq), [DEQ-Flow](https://github.com/locuslab/deq-flow), [PyTorch](https://github.com/pytorch/pytorch), and [scipy](https://github.com/scipy/scipy) for their awesome open source.