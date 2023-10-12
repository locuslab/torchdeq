# Implicit Graph Neural Networks

This repository contains the code for Implicit Graph Neural Networks (IGNN) proposed in the paper [Implicit Graph Neural Networks](https://proceedings.neurips.cc/paper/2020/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf).

## Requirements

Install dependencies via the following command.
```bash
pip install -r requirements.txt
```

## Training

To train IGNN for node classfication on the PPI dataset, run the command under `nodeclassification`.
```bash
bash run_ignn.sh
```

If you encounter training stability issues, add this flag `--norm_clip_value 0.99` to clip the re-normalization value. You can even choose a smaller one.

## Evaluation

To evaluate a pretrained model, use the folowing command.
```bash
bash run_ignn.sh --eval --resume_path [YOUR_CHECKPOINT].pth
```
You can use the `--resume_path` flag to load a model checkpoint.

We offer three checkpoints from this training setup. You can download the checkpoints from this [link](https://drive.google.com/drive/folders/1wzi5VYqKb5FPRzMlSG4A4O8XF4OV82Q0?usp=sharing). Here is the evaluation performance.

|  Checkpoint Name | PPI Val | PPI Test |
| :--------------: | :-----: | :------: |
| ignn-ppi-1.pth | 97.78 | 98.56 |
| ignn-ppi-2.pth | 97.88 | 98.65 |
| ignn-ppi-3.pth | 98.16 | 98.83 |

To download checkpoints to your local machine, copy the `FILE_ID` from each checkpoint and run this command to fetch a checkpoint.

```bash
wget https://drive.google.com/uc?id=FILE_ID
```

## Reference

```bib
@inproceedings{gu2020implicit,
 author = {Gu, Fangda and Chang, Heng and Zhu, Wenwu and Sojoudi, Somayeh and El Ghaoui, Laurent},
 booktitle = {Advances in Neural Information Processing Systems},
 title = {Implicit Graph Neural Networks},
 year = {2020}
}
```