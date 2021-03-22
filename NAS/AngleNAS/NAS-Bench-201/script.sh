export TORCH_HOME='/data/ABS/NAS-Bench-201/'

# Fig. 2
bash ./scripts-search/algos/SETN.sh ImageNet16-120 0 0

bash ./scripts-search/algos/SPOS.sh ImageNet16-120 0 0

bash ./scripts-search/algos/GDAS.sh ImageNet16-120 0 0

bash ./scripts-search/algos/ENAS.sh ImageNet16-120 0 0

bash ./scripts-search/algos/DARTS-V1.sh ImageNet16-120 0 0

# Fig 4
bash ./scripts-search/algos/train_standalone_models.sh cifar10 10

bash ./scripts-search/algos/get_standalone_ranks.sh cifar10 10

# Fig. 5, 6, table 1, 2
bash ./scripts-search/algos/train_supernet.sh cifar10 0 250

bash ./scripts-search/algos/get_angles.sh cifar10 0 249 'angle'

bash ./scripts-search/algos/get_acc.sh cifar10 0 249 'acc'

bash ./scripts-search/algos/cal_correlation.sh cifar10 0 'angle-epoch_249'

# Fig. 7
bash ./scripts-search/algos/dropnode_by_angle.sh cifar10 0 100 'dropnode_angle'

bash ./scripts-search/algos/dropnode_by_acc.sh cifar10 0 100 'dropnode_acc'

bash ./scripts-search/algos/dropnode_by_magnitude.sh cifar10 0 100 'dropnode_magnitude'

# Appendix Fig.4
bash ./scripts-search/algos/train_a_standalone_model.sh cifar10 0 100 'standalone_model'

bash ./scripts-search/algos/get_angle_evolution.sh cifar10 0 100 'standalone_model'

# visualization
python3 exps/NAS-Bench-201/visualize.py --api_path ./NAS-Bench-102-v1_0-e61699.pth