#!/bin/bash
# python PATH
export PYTHONPATH="${PYTHONPATH}:${HOME}/github"

# hyperparameter
#echo -n "input the gpu (seperate by comma (,) ): "
#read gpus
gpus=0
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
# replace comma(,) with empty
#gpus=${gpus//,/}	
# the number of characters
#num_gpus=${#gpus}
#echo "the number of gpus is ${num_gpus}"

# choose the dataset
#echo ""
echo "0  --  cifar10"
echo "1  --  cifar100"
echo "2  --  ImageNet"
echo "3  --  SVHN"
echo -n "choose the dataset: "
read dataset_choose

#dataset_choose=1
case ${dataset_choose} in
	0 )
		dataset='cifar10'
		crop_size=32
		batch_size=128
		epochs=300
		;;
	1 ) 
		dataset='cifar100'
		crop_size=32
		batch_size=128
		epochs=300
		;;
	2 )
		dataset='imagenet'
		crop_size=224
		batch_size=256
		epochs=120
		;;
	3 )
		data="${HOME}/dataset/svhn"
		dataset='svhn'
		batch_size=128
		crop_size=32
		epochs=160
		;;
	* )
		echo "The choice of the dataset is illegal!"
		exit 1 
		;;
esac

if [[ ${dataset} == 'imagenet' ]]; then
	echo ""
	echo -n "input the crop_size: "
	read crop_size
fi

# choose the arch
echo ""
echo -n "input the name of architecture: "
read arch

echo -n "choose the split factor (1, 2, 4): "
read split_factor

# set the work dir
work_dir="${HOME}/github/splitnet"
is_apex_amp=0
is_official_densenet=1
is_lukemelas_efficientnet=1
is_efficientnet_user_crop=1

# train the model
python ${work_dir}/train_split.py 	--dist_url 'tcp://127.0.0.1:6791' \
									--multiprocessing_distributed \
									--world_size 1 \
									--rank 0 \
									--gpu_ids ${gpus} \
									--dataset ${dataset} \
									--crop_size ${crop_size} \
									--batch_size ${batch_size} \
									--arch ${arch} \
									--split_factor ${split_factor} \
									--is_summary 1 \
									--is_mixup 0 \
									--epochs ${epochs} \
									--is_apex_amp ${is_apex_amp} \
									--is_official_densenet ${is_official_densenet} \
									--is_lukemelas_efficientnet ${is_lukemelas_efficientnet} \
									--is_efficientnet_user_crop ${is_efficientnet_user_crop}

echo "Summary Finished!!!"
