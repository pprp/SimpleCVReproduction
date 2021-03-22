#!/bin/bash
# Single-Path One-Shot
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset and seed"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

dataset=$1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=nas-bench-102

if [ "$dataset" == "cifar10" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar10/"
  config_path="configs/nas-benchmark/algos/SPOS_standalone_cifar10.config"
fi

if [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar100/"
  config_path="configs/nas-benchmark/algos/SPOS_standalone_cifar100.config"
fi

if [ "$dataset" == "ImageNet16-120" ]; then
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
  config_path="configs/nas-benchmark/algos/SPOS_standalone_imagenet.config"
fi

arch_idx=0 
for ((i=1; i<=50; i ++))
do
  save_dir=./output/search-cell-${space}/result-${dataset}/standalone_arch-${arch_idx}
  OMP_NUM_THREADS=4 python3 ./exps/angle/train_standalone_models.py \
    --save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
    --dataset ${dataset} --data_path ${data_path} --arch_idx ${arch_idx}\
    --search_space_name ${space} \
    --arch_nas_dataset ${TORCH_HOME}/NAS-Bench-102-v1_0-e61699.pth \
    --config_path ${config_path} \
    --track_running_stats 1 \
    --arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
    --select_num 100 \
    --workers 4 --print_freq 200 --rand_seed ${seed}
    let arch_idx+=200
done

