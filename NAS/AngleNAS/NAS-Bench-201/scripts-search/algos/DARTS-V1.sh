#!/bin/bash
# bash ./scripts-search/algos/DARTS-V1.sh cifar10 -1
# Fig. 2
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for dataset and seed"
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
space=$3 # Based on NAS-Bench-201, we design 11 shrunk search spaces of various size, whose no. ranges from 0 to 10.
channel=16
num_cells=5
max_nodes=4


if [ "$dataset" == "cifar10" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar10/"
fi

if [ "$dataset" == "cifar100" ]; then
  data_path="$TORCH_HOME/cifar.python/cifar100/"
fi

if [ "$dataset" == "ImageNet16-120" ]; then
  data_path="$TORCH_HOME/cifar.python/ImageNet16"
fi

save_dir=./output/search-cell-nas-bench-102/DARTS-V1-${dataset}

OMP_NUM_THREADS=4 python3 ./exps/algos/DARTS-V1.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path configs/nas-benchmark/algos/DARTS.config \
	--arch_nas_dataset ${TORCH_HOME}/NAS-Bench-102-v1_0-e61699.pth \
	--track_running_stats 1 \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
