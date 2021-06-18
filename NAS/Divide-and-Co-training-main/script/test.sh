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

# choose the arch
echo ""
echo -n "input the name of architecture: "
read arch
eval_batch_size=80


# set the crop size for ImageNet
if [[ ${dataset} == 'imagenet' ]]; then
	echo ""
	echo -n "input the crop_size (default: 224): "
	read crop_size
	if [ -z "${crop_size}" ];then
		crop_size=224
	fi
fi


echo ""
echo -n "choose the split factor (1, 2, 4, default: 1): "
read split_factor
if [ -z "${split_factor}" ];then
	split_factor=1
fi

echo ""
echo -n "using multi gpus test (0, 1, default: 1): "
read is_test_on_multigpus
if [ -z "${is_test_on_multigpus}" ];then
	is_test_on_multigpus=1
fi

echo ""
echo -n "using multi streams test (0, 1, default: 0): "
read is_test_with_multistreams
if [ -z "${is_test_with_multistreams}" ];then
	is_test_with_multistreams=0
fi

#echo ""
#echo -n "choose the ensemble manner (0 - avg, 1 - max, default: 0): "
#read is_max_ensemble
if [ -z "${is_max_ensemble}" ];then
	is_max_ensemble=0
fi


#echo ""
#echo -n "ensemble after softmax (0 - no, 1 - yes, default: 0): "
#read is_ensembled_after_softmax
if [ -z "${is_ensembled_after_softmax}" ];then
	is_ensembled_after_softmax=0
fi

# setting about efficientnet
is_efficientnet_user_crop=0
is_lukemelas_efficientnet=1
is_memory_efficient_swish=1

SPID=wide_resnet50_3_split1_imagenet_256_01
SPID=wide_resnet50_3_split2_imagenet_256_02
SPID=wide_resnet50_2_split1_imagenet_256_01
SPID=wide_resnet50_2_split2_imagenet_256_03
SPID=resnext101_64x4d_split1_imagenet_256_01
SPID=resnext101_64x4d_split2_imagenet_256_02
SPID=se_resnext101_64x4d_split2_imagenet_128_02
#SPID=efficientnetb7_split2_imagenet_128_02
#SPID=resnet50_split1_imagenet_256_06
#SPID=resnet101_split1_imagenet_256_01

model_dir=${HOME}/models/splitnet/${SPID}
model_dir=${HOME}/models/imagenet/${SPID}
#model_dir=${HOME}/models/resnext29/${SPID}
resume=${model_dir}/model_best.pth.tar
#resume=None
#resume=${model_dir}/checkpoint_291.pth.tar

pretrained_dir=None
pretrained_dir=${HOME}/models/pretrained/efficientnet-b5-b6417697.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b4-6ed6700e.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b3-5fb5a3c3.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b2-8bb594d6.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b1-f1951068.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b7-dcc49843.pth
pretrained_dir=${HOME}/models/pretrained/efficientnet-b6-c76e70fd.pth

# some other settings
workers=8
norm_mode=batch
is_amp=1
is_wd_all=0

# set the work dir
work_dir="${HOME}/github/splitnet"

# evaluate the model
python ${work_dir}/test.py 			--dist_url 'tcp://127.0.0.1:6000' \
									--multiprocessing_distributed \
									--world_size 1 \
									--epochs 90 \
									--rank 0 \
									--workers ${workers} \
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
									--is_max_ensemble ${is_max_ensemble} \
									--is_test_on_multigpus ${is_test_on_multigpus} \
									--is_test_with_multistreams ${is_test_with_multistreams} \
									--is_efficientnet_user_crop ${is_efficientnet_user_crop} \
									--is_lukemelas_efficientnet ${is_lukemelas_efficientnet} \
									--is_memory_efficient_swish ${is_memory_efficient_swish} \
									--pretrained_dir ${pretrained_dir}

echo "Evaluating Finished!!!"
