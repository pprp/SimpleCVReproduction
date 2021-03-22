# ABS
This project provides Pytorch implementation for [Angle-based Search Space Shrinking for Neural Architecture Search](https://arxiv.org/abs/2004.13431).

<!-- ![introduce image](image/pipeline.png) -->
<img width="740" height="370" src="figure/pipeline.png"/>

## Requirements
- Pytorch 1.3
- Python 3.5+
- [Apex](https://github.com/NVIDIA/apex)

The requirements.txt file lists other Python libraries that this project depends on, and they will be installed using:
pip3 install -r requirements.txt

## Searched Models with ABS

| Model | Flops | Top1 Acc. | Flops (ABS) | Top1 (ABS) | GoogleDrive |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  SPOS  |  465M | 75.33% | 472M	| 75.97% | [Model](https://drive.google.com/file/d/1mDgSi2LisO6OCimL3otNN8FCXvn1niEQ/view?usp=sharing)|
|  FairNAS | 322M | 74.24% | 325M | 74.42% | [Model](https://drive.google.com/file/d/1NhlkDH2TM-fBv20U45RSmGr7nDQLnDa4/view?usp=sharing)|
|  ProxylessNAS | 467M | 75.56% | 470M	| 76.14% | [Model](https://drive.google.com/file/d/1XczhRsSCXT7Ue__TikldblyoHFehkgsK/view?usp=sharing)|
|  DARTS | 530M | 74.88% | 619M (547M)	| 75.59% (75.19%) | [Model](https://drive.google.com/file/d/1mOC1g7NAzSazg9yFFwWyyVLj9or6dJG2/view?usp=sharing), [Scale Down](https://drive.google.com/file/d/19XJJT6-N3leZzsDwh5Jlbhczoo4fkcuX/view?usp=sharing)|
|  PDARTS | 553M | 75.58% | 645M (570M)| 75.89% (75.64%) | [Model](https://drive.google.com/file/d/1xSurlv5bzQ4rTIIbueoe8qhigW1X00dS/view?usp=sharing), [Scale Down](https://drive.google.com/file/d/1nfomE9euuBWhNlwGa-jp3ldczVnTp1cz/view?usp=sharing)|

For the form x(y), x means models searched without human intervention, y means the models whose channels are scaled down to fit with the constraint of flops

## Usage
### Step 1: Setup Dataset
We have splitted 50000 images from `ImageNet Train Dataset` as the validation set for search. The remainings are used for supernet training.

Run `utils/get_flops_lookup_table.sh` to generate flops lookup table which is required in Uniform Sampling.

### Step 2: Search Space Shrinking
Shrink search spaces with the following command:
```
cd shrinking
python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
                                    --train_dir YOUR_TRAINDATASET_PATH
```
Note: SPOS and ProxylessNAS share the same shrunk search space. DARTS and PDARTS share the same shrunk search space

### Step 3: Search over the Shrunk Search Space
#### DARTS:
```
cd darts-master
python3 train_search.py --data $YOUR_DATA_PATH --unrolled --save DARTS_ABS \
						--operations_path $YOUR_SHRUNK_SEARCH_SPACE
```

#### PDARTS:
```
cd pdarts-master
python3 train_search.py --save PDARTS_ABS --tmp_data_dir $YOUR_DATA_PATH \
						--operations_path $YOUR_SHRUNK_SEARCH_SPACE

```

#### ProxylessNAS:
```
cd searching
python3 imagenet_arch_search.py --path ABS 
                                --target_hardware flops \
                                --operations_path $YOUR_SHRUNK_SEARCH_SPACE \
                                --train_dir $YOUR_TRAINDATASET_PATH --test_dir $YOUR_TESTDATASET_PATH
```

#### SPOS and FairNAS search with the following procedures:

##### setup a server for the distributed search
```
tmux new -s mq_server
sudo apt update
sudo apt install rabbitmq-server
sudo service rabbitmq-server start
sudo rabbitmqctl add_user test test
sudo rabbitmqctl set_permissions -p / test '.*' '.*' '.*'
```

##### train and search
Before search, please modify host and username in the config file searching/config.py.
```
cd searching
python3 -m torch.distributed.launch --nproc_per_node=8 main.py --operations_path \
                            --train_dir $YOUR_TRAINDATASET_PATH --test_dir $YOUR_TESTDATASET_PATH
$YOUR_SHRUNK_SEARCH_SPACE
```

##### start new tmuxs for model evaluation (concurrent with last Step)
```
tmux new -s server_x
cd searching
python3 test_server.py
```
You can start more than one test_server.py to speed up, if you have enough **GPUs** and **memory** researces.

### Step 4: Train from Scratch

Finally, train and evaluate the searched architectures with the following command.

Train:

```
cd training
python3 -m torch.distributed.launch --nproc_per_node=8 train_from_scratch.py \
                            --train_dir $YOUR_TRAINDATASET_PATH --test_dir $YOUR_TESTDATASET_PATH
```

Evaluate:

```
cd training
python3 -m torch.distributed.launch --nproc_per_node=8 train_from_scratch.py \
                            --eval --eval-resume $YOUR_WEIGHT_PATH \
                            --train_dir $YOUR_TRAINDATASET_PATH --test_dir $YOUR_TESTDATASET_PATH
```

## Thanks
This implementation of ABS is based on [DARTS](https://github.com/quark0/darts), [PDARTS](https://github.com/chenxin061/pdarts), [ProxylessNAS](https://github.com/mit-han-lab/ProxylessNAS), [SPOS](https://github.com/megvii-model/SinglePathOneShot), [NAS-Benchmark-201](https://github.com/D-X-Y/AutoDL-Projects). Except to replace the search space, everything else follows their original implementation. Please ref to their reposity for more details.

## Citation
If you find that this project helps your research, please consider citing some of the following papers:

```
@article{hu2020abs,
    title={Angle-based Search Space Shrinking for Neural Architecture Search},
    author={Yiming Hu, Yuding Liang, Zichao Guo, Ruosi Wan, Xiangyu Zhang, Yichen Wei, \
    	Qingyi Gu, Jian Sun},
    year={2020},
    booktitle = {arXiv preprint arXiv:2004.13431},
}
```

```
@article{guo2019single,
        title={Single path one-shot neural architecture search with uniform sampling},
        author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, \
        	Zechun and Wei, Yichen and Sun, Jian},
        journal={arXiv preprint arXiv:1904.00420},
        year={2019}
}
```
