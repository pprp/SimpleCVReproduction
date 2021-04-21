# ============================= mobilenet v2 ilsvrc decode  ============================
# This is the setting on 8 NVidia-V100.

python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
  --gpu_id 0,1,2,3,4,5,6,7 \
  --port 6720 \
  --print_freq 100 \
  --exec_mode train \
  --learner trick \
  --dataset ilsvrc_12 \
  --data_path ~/datasets/imagenet_origin \
  --model_type mobilenet_v2_decode \
  --batch_size 512 \
  --lr 2e-1 \
  --lr_min 0. \
  --lr_decy_type cosine \
  --weight_decay 4e-5 \
  --nesterov \
  --label_smooth \
  --label_smooth_eps 1e-1 \
  --dropout_rate 0.0 \
  --cfg 40,40,20,60,60,30,72,72,30,90,90,40,120,120,40,120,120,40,96,96,80,192,192,80,144,144,80,240,240,80,480,480,120,432,432,120,576,576,120,432,432,200,1200,1200,200,960,960,200,1200,1200,400,1600 \
  --eval_epoch 5 \
  --epochs 250

# Searched MobileNet-2 Architectures (--cfg):
# 1) mobilenet-v2 314M: 40,40,20,60,60,30,72,72,30,90,90,40,120,120,40,120,120,40,96,96,80,192,192,80,144,144,80,240,240,80,480,480,120,432,432,120,576,576,120,432,432,200,1200,1200,200,960,960,200,1200,1200,400,1600
# 2) mobilenet-v2 156M: 24,24,12,36,36,24,54,54,24,90,90,24,72,72,24,72,72,24,72,72,64,192,192,64,96,96,64,96,96,64,192,192,48,288,288,48,576,576,48,576,576,160,960,960,160,720,720,160,960,960,240,1280