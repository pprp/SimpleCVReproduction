# ============================= resnet18 ilsvrc decode  ============================
# This is the setting on 8 NVidia-V100.

python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
  --gpu_id 0,1,2,3,4,5,6,7 \
  --port 6657 \
  --print_freq 100 \
  --exec_mode train \
  --learner trick \
  --dataset ilsvrc_12 \
  --data_path ~/datasets/imagenet_origin \
  --model_type resnet_18_decode \
  --batch_size 512 \
  --lr 0.1 \
  --lr_min 0. \
  --lr_decy_type cosine \
  --weight_decay 1e-4 \
  --nesterov \
  --cfg 48,64,64,64,48,160,128,96,96,320,256,256,256,512,512,640,640 \
  --eval_epoch 5 \
  --epochs 120

# Searched ResNet-18 Architectures (--cfg):
# 1) resnet-18 1.83G: 48,64,64,64,48,160,128,96,96,320,256,256,256,512,512,640,640
# 2) resnet-18 1.05G: 32,48,48,48,64,96,64,96,96,192,192,192,192,384,384,384,640
