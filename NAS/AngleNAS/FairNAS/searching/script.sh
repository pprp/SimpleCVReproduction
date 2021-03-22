python3 -m torch.distributed.launch --nproc_per_node=8 main.py --operations_path '../shrinking/shrunk_search_space.pt' | tee -a searching.log

python3 test_server.py | tee -a testing.log