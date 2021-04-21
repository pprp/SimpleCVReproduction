# ============================= mobilenet_v2 ilsvrc search  ============================
# This is the setting on 8 NVidia-V100.
# For other variants of searching, please refer to start_search_resnet_cifar.sh

python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
  --gpu_id 0,1,2,3,4,5,6,7 \
  --seed 1 \
  --fix_random \
  --port 6672 \
  --exec_mode train \
  --print_freq 100  \
  --learner chann_ilsvrc \
  --dataset ilsvrc_12 \
  --data_path ~/datasets/imagenet_origin \
  --model_type mobilenet_v2_width \
  --blockwise \
  --batch_size 2048 \
  --lr 5e-2 \
  --lr_decy_type cosine \
  --weight_decay 4e-5 \
  --arch_weight_decay 1e-3 \
  --arch_learning_rate 1.6e-4 \
  --entropy_coeff 4e-1 \
  --flops \
  --max_flops 3.2e8 \
  --flops_coeff -0 -0.1 \
  --flops_dir ./mobilenet_v2_flops.pkl \
  --ft_schedual 'fixed' \
  --ft_proj_lr 1e-4 \
  --updt_proj 10 \
  --norm_constraint 'constraint' \
  --orthg_weight 5e-3 \
  --norm_weight 1e-2 \
  --multiplier 1 \
  --max_width 32 \
  --candidate_width 8,12,16,20 \
  --beam_search \
  --top_seq 8 \
  --eval_epoch 5 \
  --warmup_epochs 80 \
  --epochs 160

