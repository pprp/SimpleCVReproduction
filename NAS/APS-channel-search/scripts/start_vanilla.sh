# NOTE: THIS PART IS NOT NECESSARY FOR APS-CHANNEL-SEARCH
# This is the script for running vanilla baselines.
# Uncomment the ones that you need to run.

## ========================= resnet20 at cifar10  =========================
# python main.py \
#   --seed 0 \
#   --gpu_id $1 \
#   --exec_mode train \
#   --print_freq 30 \
#   --learner vanilla \
#   --dataset cifar10 \
#   --data_path ~/datasets \
#   --model_type resnet_20\
#   --eval_epoch 1 \
#   --epochs 200 

## ========================= resnet18 at ilsvrc12 =========================
# python -u -m torch.distributed.launch --nproc_per_node=2 main.py \
#   --seed 0 \
#   --fix_random \
#   --gpu_id $1 \
#   --exec_mode train \
#   --print_freq 30 \
#   --learner vanilla \
#   --dataset ilsvrc_12 \
#   --data_path ~/datasets/imagenet_origin \
#   --model_type resnet_18\
#   --batch_size 1024 \
#   --eval_epoch 1 \
#   --epochs 100


## ======================= mobilenet_v2 at ilsvrc =========================
# python -u -m torch.distributed.launch --nproc_per_node=4 main.py \
#   --seed 0 \
#   --fix_random \
#   --gpu_id $1 \
#   --exec_mode train \
#   --print_freq 20 \
#   --learner vanilla \
#   --dataset ilsvrc_12 \
#   --data_path ~/datasets/imagenet_origin \
#   --model_type mobilenet_v2\
#   --dropout_rate 0. \
#   --weight_decay 4e-5 \
#   --lr 5e-2 \
#   --batch_size 256 \
#   --eval_epoch 5 \
#   --epochs 150


