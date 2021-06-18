#!/bin/bash
# python PATH
export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
# replace comma(,) with empty
#gpus=${gpus//,/}	
# the number of characters
#num_gpus=${#gpus}
#echo "the number of gpus is ${num_gpus}"

data_dir=${HOME}
#data_dir=/dev/shm

# choose the dataset
echo ""
echo "0  --  cifar10"
echo "1  --  cifar100"
echo "2  --  ImageNet"
echo "3  --  svhn"
echo -n "choose the dataset: "
read dataset_choose

cot_weight_warm_up_epochs=40
case ${dataset_choose} in
	0 )
		data="${data_dir}/dataset/cifar"
		dataset=cifar10
		batch_size=128
		crop_size=32
		epochs=300
		;;
	1 ) 
		data="${data_dir}/dataset/cifar"
		dataset=cifar100
		batch_size=128
		crop_size=32
		epochs=300
		;;
	2 )
		data="${HOME}/dataset/imagenet"
		dataset=imagenet
		batch_size=256
		crop_size=224
		epochs=120
		cot_weight_warm_up_epochs=60
		;;
	3 )
		data="${HOME}/dataset/svhn"
		dataset=svhn
		batch_size=128
		crop_size=32
		epochs=200
		;;
	* )
		echo "The choice of the dataset is illegal!"
		exit 1 
		;;
esac


# choose architecture
# CIFAR: resnet110, resnet164, resnext29_8x64d
# ImageNet: resnet50, resnet110, efficientnetb1, etc
arch=resnet50
is_amp=1

workers=16
is_syncbn=0

# parameters of splitnet
split_factor=1
cot_weight=0.5
is_cot_loss=1
cot_loss_choose='js_divergence'
is_diff_data_train=1
is_cot_weight_warm_up=1
is_div_wd=0

# 'cos', 'step', 'poly', 'HTD'
lr_mode=cos
is_cutout=1
erase_p=0.5
is_mixup=1
is_autoaugment=1

# setting about efficientnet
is_efficientnet_user_crop=0
is_lukemelas_efficientnet=1
is_memory_efficient_swish=0

# distributed training
world_size=1
rank=0
dist_url='tcp://127.0.0.1:6066'
optimizer=SGD

# set the work dir
work_dir="${HOME}/github/splitnet"
resume=None
iters_to_accumulate=1

for num in 07
do
	case ${num} in
		00 )
			arch=se_resnet50
			split_factor=1
			;;	
		07 )
			arch=se_resnet50_B
			#resume=${HOME}/models/splitnet/se_resnet50_B_split1_imagenet_256_06/checkpoint_37.pth.tar
			;;
		12 )
			arch=efficientnetb1
			lr_mode=exponential
			optimizer=RMSpropTF
			split_factor=1
			is_efficientnet_user_crop=0
			resume=${HOME}/models/splitnet/efficientnetb1_split1_imagenet_256_12/checkpoint_106.pth.tar
			;;
		* )
			;;					
	esac

	# model directory
	SPID="${arch}_split${split_factor}_${dataset}_${batch_size}_${num}"
	model_dir=${HOME}/models/splitnet/${SPID}
	proc_name=${SPID}

	# detect the directory
	if [ -d ${model_dir} ]
	then
		echo "save model into ${model_dir}"
	else
		mkdir ${model_dir}
		echo "make the directory ${model_dir}"
	fi

	# train the model
	python ${work_dir}/train_split.py 	--dist_url ${dist_url} \
										--multiprocessing_distributed \
										--world_size ${world_size} \
										--rank ${rank} \
										--gpu_ids ${gpus} \
										--data ${data} \
										--resume ${resume} \
										--dataset ${dataset} \
										--crop_size ${crop_size} \
										--batch_size ${batch_size} \
										--model_dir ${model_dir} \
										--arch ${arch} \
										--proc_name ${SPID} \
										--split_factor ${split_factor} \
										--is_cot_loss ${is_cot_loss} \
										--cot_weight ${cot_weight} \
										--is_diff_data_train ${is_diff_data_train} \
										--is_cutout ${is_cutout} \
										--erase_p ${erase_p} \
										--lr_mode ${lr_mode} \
										--is_mixup ${is_mixup} \
										--epochs ${epochs} \
										--workers ${workers} \
										--cot_loss_choose ${cot_loss_choose} \
										--is_autoaugment ${is_autoaugment} \
										--is_cot_weight_warm_up ${is_cot_weight_warm_up} \
										--is_syncbn ${is_syncbn} \
										--is_div_wd ${is_div_wd} \
										--cot_weight_warm_up_epochs ${cot_weight_warm_up_epochs} \
										--is_amp ${is_amp} \
										--is_efficientnet_user_crop ${is_efficientnet_user_crop} \
										--iters_to_accumulate ${iters_to_accumulate} \
										--is_lukemelas_efficientnet ${is_lukemelas_efficientnet} \
										--is_memory_efficient_swish ${is_memory_efficient_swish} \
										--optimizer ${optimizer}

done

echo "Training Finished!!!"
