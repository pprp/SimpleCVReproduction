python3 -m torch.distributed.launch --nproc_per_node=8 train_from_scratch.py --auxiliary --save='DARTS_ABS' --arch='DARTS_ABS' --init_channels 45

python3 train_cifar.py --save DARTS_ABS_CIFAR --arch='DARTS_ABS' --tmp_data_dir /data/cifar10 --init_channels 16 --layers 8 --auxiliary