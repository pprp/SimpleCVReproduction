## ============================= resnet cifar10 decode  ============================
python main.py \
  --gpu_id 0 \
  --exec_mode train \
  --learner vanilla \
  --dataset cifar10 \
  --data_path ~/datasets \
  --model_type resnet_decode \
  --lr 0.1 \
  --lr_min 0. \
  --cfg 8,8,8,8,8,8,8,16,16,16,16,32,32,32,64,64,64,64,64 \
  --lr_decy_type cosine \
  --weight_decay 5e-4 \
  --nesterov \
  --epochs 300

# Searched ResNet-20 Architectures (--cfg):
# 1) resnet-20 20.6M: 8,8,8,8,8,8,8,16,16,16,16,32,32,32,64,64,64,64,64
# 2) resnet-56 60.3M: 16,16,16,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,64,64,64,64,64,64,64,64,64,64,64,64             