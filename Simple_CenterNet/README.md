# Pytorch simple CenterNet-45

This repository is a simple pytorch implementation of [Objects as Points](https://arxiv.org/abs/1904.07850), some of the code is taken from the [official implementation](https://github.com/xingyizhou/CenterNet).
As the name says, this version is **simple** and **easy to read**, all the complicated parts (dataloader, hourglass, training loop, etc) are all rewrote in a simpler way.    
By the way the support of **nn.parallel.DistributedDataParallel** is also added, so this implementation trains significantly faster than the official code (~ **75 img/s** vs ~36 img/s on 8 GPUs).

Enjoy!     
 
## Requirements:
- python>=3.5
- pytorch==0.4.1 or 1.1.0 (DistributedDataParallel training only available using 1.1.0)
- tensorboardX(optional)

## Getting Started
1. Disable cudnn batch normalization.
Open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`.

2. Clone this repo:
    ```
    CenterNet_ROOT=/path/to/clone/CenterNet
    git clone https://github.com/zzzxxxttt/pytorch_simple_CenterNet_45 $CenterNet_ROOT
    ```

3. Install COCOAPI (the cocoapi in this repo is modified to work with python3):
    ```
    cd $CenterNet_ROOT/lib/cocoapi/PythonAPI
    make
    python setup.py install --user
    ```

4. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2)).
    If you are using pytorch 0.4.1, rename ```$CenterNet_ROOT/lib/DCNv2_old``` to ```$CenterNet_ROOT/lib/DCNv2```, otherwise rename ```$CenterNet_ROOT/lib/DCNv2_new``` to ```$CenterNet_ROOT/lib/DCNv2```.
    ```
    cd $CenterNet_ROOT/lib/DCNv2
    ./make.sh
    ```

5. Compile NMS.
    ```
    cd $CenterNet_ROOT/lib/nms
    make
    ```

6. For COCO training, Download [COCO dataset](http://cocodataset.org/#download) and put ```annotations```, ```train2017```, ```val2017```, ```test2017``` (or create symlinks) into ```$CenterNet_ROOT/data/coco```

7. For Pascal VOC training, download [VOC0712 in coco format (password: 4iu2)](https://pan.baidu.com/s/1z6BtsKPHh2MnbfT25Y4wYw) and put ```annotations```, ```images```, ```VOCdevkit``` (or create symlinks) into ```$CenterNet_ROOT/data/voc```

8. To train Hourglass-104, download [CornerNet pretrained weights (password: y1z4)](https://pan.baidu.com/s/1tp9-5CAGwsX3VUSdV276Fg) and put ```checkpoint.t7``` into ```$CenterNet_ROOT/ckpt/pretrain```.


## Train 
### COCO
#### single GPU or multi GPU using nn.DataParallel
```
python train.py --log_name coco_hg_512_dp \
                --dataset coco \
                --arch large_hourglass \
                --lr 5e-4 \
                --lr_step 90,120 \
                --batch_size 48 \
                --num_epochs 140 \  
                --num_workers 10
```
#### multi GPU using nn.parallel.DistributedDataParallel
```
python -m torch.distributed.launch --nproc_per_node NUM_GPUS train.py --dist \
        --log_name coco_hg_512_ddp \
        --dataset coco \
        --arch large_hourglass \
        --lr 5e-4 \
        --lr_step 90,120 \
        --batch_size 48 \
        --num_epochs 140 \
        --num_workers 2
```

### PascalVOC
#### single GPU or multi GPU using nn.DataParallel
```
python train.py --log_name pascal_resdcn18_384_dp \
                --dataset pascal \
                --arch resdcn_18 \
                --img_size 384 \
                --lr 1.25e-4 \
                --lr_step 45,60 \
                --batch_size 32 \
                --num_epochs 70 \
                --num_workers 10
```
#### multi GPU using nn.parallel.DistributedDataParallel
```
python -m torch.distributed.launch --nproc_per_node NUM_GPUS train.py --dist \
        --log_name pascal_resdcn18_384_ddp \
        --dataset pascal \
        --arch resdcn_18 \
        --img_size 384 \
        --lr 1.25e-4 \
        --lr_step 45,60 \
        --batch_size 32 \
        --num_epochs 70 \
        --num_workers 2
```
## Evaluate
### COCO
```
python test.py --log_name coco_hg_512_dp \
               --dataset coco \
               --arch large_hourglass

# flip test
python test.py --log_name coco_hg_512_dp \
               --dataset coco \
               --arch large_hourglass \
               --test_flip

# multi scale test
python test.py --log_name coco_hg_512_dp \
               --dataset coco \
               --arch large_hourglass \
               --test_flip \
               --test_scales 0.5,0.75,1,1.25,1.5
```
### PascalVOC
```
python test.py --log_name pascal_resdcn18_384_dp \
               --dataset pascal \
               --arch resdcn_18 \
               --img_size 384

# flip test
python test.py --log_name pascal_resdcn18_384_dp \
               --dataset pascal \
               --arch resdcn_18 \
               --img_size 384 \
               --test_flip
```

## Results:

### COCO:
Model|Training image size|mAP
:---:|:---:|:---:
Hourglass-104 (DP)|512|39.9/42.3/45.0
Hourglass-104 (DDP)|512|40.5/42.6/45.3

### PascalVOC:
Model|Training image size|mAP
:---:|:---:|:---:
ResDCN-18 (DDP)|384|71.19/72.99
ResDCN-18 (DDP)|512|72.76/75.69


