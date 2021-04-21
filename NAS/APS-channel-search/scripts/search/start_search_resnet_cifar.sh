## ============================= resnet cifar10 learner  ============================
# NOTES:
# 1. This is the setting on a single GPU;
# 2. To perform searching without FLOPs constraint, remove --flops
# 3. To perform search with other FLOPs constraint, change --max_flops.
# 4. You can use `--beam_search` with `--top_seq N` to greedily decode top-N models from the controller;
#    Alternatively, you can use `--n_test_archs N` to randomly sample N architectures.

python main.py \
  --gpu_id 0 \
  --seed 1 \
  --fix_random \
  --exec_mode train \
  --print_freq 20  \
  --learner chann_cifar \
  --lr_decy_type multi_step \
  --dataset cifar10 \
  --data_path ~/datasets \
  --model_type resnet_20_width \
  --flops_dir ./flops_table_thinner.pkl \
  --lr 1e-1 \
  --batch_size 256 \
  --controller_type 'ENAS' \
  --arch_weight_decay 1e-3 \
  --arch_learning_rate 1.6e-4 \
  --entropy_coeff 4e-3 \
  --ft_schedual 'fixed' \
  --orthg_weight 5e-3 \
  --ft_proj_lr 0.001 \
  --flops \
  --flops_coeff -0 -0.1 \
  --max_flops 2.05e7 \
  --updt_proj 1 \
  --multiplier 1.0 \
  --max_width 124 \
  --candidate_width 4,8,16,32,64 \
  --overlap 1.0 \
  --n_test_archs 10 \
  --norm_constraint 'constraint' \
  --warmup_epochs 200 \
  --eval_epoch 5 \
  --epochs 600