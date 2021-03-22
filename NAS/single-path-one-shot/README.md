# Single Path One-Shot
https://github.com/megvii-model/SinglePathOneShot

Single Path One-Shot by Megvii Research.

## Introduction
This repository provides the implementation of [Single Path One-Shot Neural Architecture Search with Uniform Sampling](https://arxiv.org/abs/1904.00420).

## Our Trained Model / Checkpoint

+ OneDrive: [Link](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN)

### Supernet

Our trained Supernet weight is in `$Link/Supernet/checkpoint-150000.pth.tar`, which can be used by Search.

### Search

Our search result is in `$Link/Search/checkpoint.pth.tar`, which can be used by Evaluation.

### Evaluation

Out searched models have been trained from scratch, is can be found in `$Link/Evaluation/$ARCHITECTURE`.

Here is a summary:

|    Architecture         |  FLOPs    |   #Params |   Top-1   |   Top-5   |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
(2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)        |   323M     |	3.5M    |      25.6    |       8.0   |


## Usage

### 1. Setup Dataset and Flops Table

Download the ImageNet Dataset and move validation images to labeled subfolders. To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

Download the flops table to accelerate Flops calculation which is required in Uniform Sampling. It can be found in `$Link/op_flops_dict.pkl`.

We recommend to create a folder `data` and use it in both Supernet training and Evaluation training.

Here is a example structure of `data`:

```
data
|--- train                 ImageNet Training Dataset
|--- val                   ImageNet Validation Dataset
|--- op_flops_dict.pkl     Flops Table
```

### 2. Train Supernet

Train supernet with the following command:

```bash
cd src/Supernet
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

### 3. Search in Supernet with Evolutionary Algorithm

Search in supernet with the following command:

```bash
cd src/Search
python3 search.py
```

It will use ```../Supernet/checkpoint-latest.pth.tar``` as Supernet's weight, please make sure it exists or modify the path manually.

### 4. Get Searched Architecture

Get searched architecture with the following command:

```bash
cd src/Evaluation
python3 eval.py
```

It will generate folder in ``data/$YOUR_ARCHITECTURE``. You can train the searched architecture from scratch in the folder.

### 5. Train from Scratch

Finally, train and evaluate the searched architecture with the following command.

Train:

```bash
cd src/Evaluation/data/$YOUR_ARCHITECTURE
python3 train.py --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```

Evaluate:

```bash
cd src/Evaluation/data/$YOUR_ARCHITECTURE
python3 train.py --eval --eval-resume $YOUR_WEIGHT_PATH --train-dir $YOUR_TRAINDATASET_PATH --val-dir $YOUR_VALDATASET_PATH
```


## Citation

If you use these models in your research, please cite:

```
@article{guo2019single,
        title={Single path one-shot neural architecture search with uniform sampling},
        author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
        journal={arXiv preprint arXiv:1904.00420},
        year={2019}
}
```