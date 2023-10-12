# Get Started

## Installation

- Through pip.

    ```bash
    pip install torchdeq
    ```

- Through conda.

    ```bash
    conda install torchdeq
    ```


- From source.

    ```bash
    git clone https://github.com/locuslab/torchdeq.git && cd torchdeq
    pip install -e .
    ```

## Quick Start

- Automatic arg parser decorator. You can call this function to add commonly used DEQ args to your program. See the explanation for args [here](https://github.com/locuslab/torchdeq/blob/main/torchdeq/utils/arg_utils.py)!

```Python
add_deq_args(parser)
```

- Automatic DEQ definition. Call `get_deq` to get your DEQ module! It's highly decoupled implementation agnostic to your model design!

```Python
self.deq = get_deq(args)
```

- Automatic normalization for DEQ. You now do not need to add normalization manually to each weight in your DEQ module!

```Python
apply_norm(self.deq_layers)
```

- Easy DEQ forward. Even for a multi-equilibria system, you can call your DEQ in a single line!

```Python
# Assume f is a functioin of three variable a, b, c.
def fn(a, b, c):
    # Do something here...
    # Having the same input and output tensor shapes.
    return a, b, c

# A callable object (`fn` here) that defines your fixed point system.
# `fn` can be a functor defined in your Pytorch forward function.
# A functor can take your input injection from the local variables. 
# You can also pass a Pytorch Module into the DEQ class.
z_out, info = self.deq(fn, (a0, b0, c0))
```

- Automatic DEQ backward. Gradients (both exact and inexact grad) are tracked automatically! The DEQ class can sample the convergence trajectory for addition operation/supervision. Just post-process ``z_out`` as you want!

## Sample Code

```Python
import argparse

import torch

from torchdeq import get_deq, apply_norm, reset_norm
from torchdeq.utils import add_deq_args

from .layers import Injection, DEQFunc, Decoder

class DEQDemo(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.deq_func = DEQFunc(args)
        apply_norm(self.deq_func, args=args)
        self.deq = get_deq(args)

    def forward(self, x, z0):
        reset_norm(self.deq_func)
        f = lambda z: self.deq_func(z, x)
        return self.deq(f, z0)

def train(args, inj, deq, decoder, loader, loss, opt):
    for x, y in loader:
        z0 = torch.randn(args.z_shape)
        z_out, info = deq(inj(x), z0)
        l = loss(decoder(z_out[-1]), y)
        l.backward()
        opt.step()
        logger.info(f'Loss: {l.item()}, '  
          +f'Rel: {info['rel_lowest'].mean().item()}'
          +f'Abs: {info['abs_lowest'].mean().item()}')

'''Add other arguments.'''
parser = argparse.ArgumentParser()
add_deq_args(parser)
args = parser.parse_args()

inj = Injection(args)
deq = DEQDemo(args)
decoder = Decoder(args)

''' Set up loader, logger, loss, opt, etc as in standard PyTorch. '''
train(args, inj, deq, decoder, loader, loss, opt)
```