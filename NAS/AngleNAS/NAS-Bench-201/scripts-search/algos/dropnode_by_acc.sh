#!/bin/bash
# Single-Path One-Shot
echo script name: $0
echo $# arguments
if [ "$#" -ne 4 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 4 parameters for dataset and seed"
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
epochs=$3
exp_name=$4
channel=16
num_cells=5
max_nodes=4
space=nas-bench-102

if [ "$dataset" == "cifar10" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar10/"
fi

if [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar100/"
fi

if [ "$dataset" == "ImageNet16-120" ]; then
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/test-cell-${space}/result-${dataset}/${exp_name}

OMP_NUM_THREADS=4 python3 ./exps/angle/dropnode_by_acc.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} --epochs ${epochs}\
	--search_space_name ${space} \
	--arch_nas_dataset ${TORCH_HOME}/NAS-Bench-102-v1_0-e61699.pth \
	--config_path configs/nas-benchmark/algos/SPOS.config \
	--track_running_stats 1 \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--select_num 100 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
