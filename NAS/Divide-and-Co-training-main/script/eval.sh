#!/bin/bash
# python PATH
export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

data_dir=${HOME}

# choose the dataset
echo ""
echo "0  --  cifar10"
echo "1  --  cifar100"
echo "2  --  ImageNet"
echo "3  --  svhn"
echo -n "choose the dataset: "
read dataset_choose
case ${dataset_choose} in
	0 )
		data="${data_dir}/dataset/cifar"
		dataset='cifar10'
		batch_size=128
		eval_batch_size=100
		crop_size=32
		;;
	1 ) 
		data="${data_dir}/dataset/cifar"
		dataset='cifar100'
		batch_size=128
		eval_batch_size=100
		crop_size=32
		;;
	2 )
		data="${HOME}/dataset/imagenet"
		dataset='imagenet'
		batch_size=256
		eval_batch_size=40
		crop_size=224
		;;
	3 )
		data="${HOME}/dataset/svhn"
		dataset='svhn'
		batch_size=128
		crop_size=32
		;;
	* )
		echo "The choice of the dataset is illegal!"
		exit 1 
		;;
esac


# set the crop size for ImageNet
if [[ ${dataset} == 'imagenet' ]]; then
	echo ""
	echo -n "input the crop_size: "
	read crop_size
fi

# choose the arch
echo ""
echo -n "input the name of architecture: "
read arch

echo ""
echo -n "choose the split factor (1, 2, 4): "
read split_factor

echo ""
echo -n "choose the ensemble manner (0 - avg, 1 - max): "
read is_max_ensemble

echo ""
echo -n "ensemble after softmax (0 - no, 1 - yes): "
read is_ensembled_after_softmax

SPID=wide_resnet50_3_split1_imagenet_256_01
SPID=wide_resnet50_3_split2_imagenet_256_02

model_dir=${HOME}/models/splitnet/${SPID}
model_dir=${HOME}/models/resnext29/${SPID}
model_dir=${HOME}/models/imagenet/${SPID}
resume=${model_dir}/model_best.pth.tar


# some other settings
norm_mode="batch"
is_amp=1
is_wd_all=0

# set the work dir
work_dir="${HOME}/github/splitnet"

# evaluate the model
python ${work_dir}/train_split.py 	--dist_url 'tcp://127.0.0.1:6000' \
									--multiprocessing_distributed \
									--world_size 1 \
									--epochs 100 \
									--rank 0 \
									--gpu_ids ${gpus} \
									--data ${data} \
									--dataset ${dataset} \
									--model_dir ${model_dir} \
									--arch ${arch} \
									--print_freq 10 \
									--batch_size ${batch_size} \
									--eval_batch_size ${eval_batch_size} \
									--proc_name ${SPID} \
									--norm_mode ${norm_mode} \
									--evaluate \
									--resume ${resume} \
									--split_factor ${split_factor} \
									--is_ensembled_after_softmax ${is_ensembled_after_softmax} \
									--is_amp ${is_amp} \
									--crop_size ${crop_size} \
									--is_wd_all ${is_wd_all} \
									--is_max_ensemble ${is_max_ensemble}

echo "Evaluating Finished!!!"
