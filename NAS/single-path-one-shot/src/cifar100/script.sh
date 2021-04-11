# sudo mount -t tmpfs -o size=2G tmpfs /home/stack/dpj/cifar100/data
# 挂载2G内存到data文件夹中，然后传输文件到文件中
# tmpfs代表临时文件系统，是内存中的一块空间

CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_fair_way.py 