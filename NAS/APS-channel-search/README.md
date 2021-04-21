# Affine Parameter Sharing for Channel Search

This repository contains the implementation of our paper ["Revisiting Parameter Sharing for Automatic Neural Channel Number Search"](https://proceedings.neurips.cc/paper/2020/file/42cd63cb189c30ed03e42ce2c069566c-Paper.pdf) in NeurIPS 2020.
We propose affine parameter sharing as a general framework to quantitatively analyze and better utilize parameter sharing for channel number search problems.
An overall illustration is presented below:
![model](./assets/model.png)

## Dependencies
Our implementation is built on Python 3.6 with Pytorch. Install packages: 
```bash
pip install -r requirements.txt
```

## Execution 
### 1: Prepare datasets
Our experiments are based on CIFAR-10 and ImageNet-2012. Please first download the datasets yourself.
The default path to dataset is set to `--data_path TODO` in `./main.py`. Please change this in the follwowing execution script correspondingly.

### 2: Searching
The searching scripts are in `./scripts/search`.
* The default parameters are set for each model in the scripts. 
* For Imagenet experiment, we use torch.distributed for parallel training. Make sure the number of GPU used matches `--nproc_per_node`. For example:
`python -u -m torch.distributed.launch --nproc_per_node=4 main.py  --gpu_id 0,1,2,3`
* To perform searching without FLOPs constraint, remove `--flops`; You can also perform the search under your desired FLOPs,
by changing `--max_flops ${DESIRED_FLOPs}`.
* More configurations can be found in the excuting scripts.

### 3: Training from scratch
The scripts are in `./scripts/train/`.
* The model configurations with different FLOPs are provided. Please uncomment the one that you need to run.

More descriptions of script args can be found in `./main.py`.



## Model Checkpoints & Training Logs
We also provide model checkpoints and training logs on CIFAR-10 and ImageNet datasets [here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155102334_link_cuhk_edu_hk/Ev1VrShV_EROlM3RYOc5Ue8BVLolBg9j1kmecmx7PGYYSw?e=g1aaAV (`./logs`)).
Note that the repository is minorly refacted from the original implementation in the paper, and the above procedures may not gaurantee to reproduce the same results due to searching randomness. 
However, the searched models are roughly close in general with similar accuracies.


## Citation
If you find this repo helpful for your research, please: 
```
@inproceedings{wang2020revisiting,
  title={Revisiting Parameter Sharing for Automatic Neural Channel Number Search},
  author={Wang, Jiaxing and Bai, Haoli and Wu, Jiaxiang and Shi, Xupeng and Huang, Junzhou and King, Irwin and Lyu, Michael and Cheng, Jian},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
