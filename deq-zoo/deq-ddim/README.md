# Deep Equilibrium Approaches to Diffusion Models

This repository contains the code for DEQ-DDIM proposed in the paper [Deep Equilibrium Approaches to Diffusion Models](https://arxiv.org/abs/2210.12867). 

## Requirements

Install dependencies via the following command.
```bash
conda create --name [ENV_NAME] --file requirements.txt
conda activate [ENV_NAME]
```

## Image Inversion

To invert images back to the latent space of diffusion models, run the command.
```bash
bash scripts/invert_models_deq.sh
```

## Reference

```bib
@article{DEQ-DDIM,
  title={Deep Equilibrium Approaches to Diffusion Models},
  author={Pokle, Ashwini and Geng, Zhengyang and Kolter, J Zico},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{geng2021training,
    title = {On Training Implicit Models},
    author = {Zhengyang Geng and Xin-Yu Zhang and Shaojie Bai and Yisen Wang and Zhouchen Lin},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2021}
}
```