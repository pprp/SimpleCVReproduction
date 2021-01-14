# ENAS_micro_retrain_Pytorch
## Training
`python train_enas_cnn_network.py`

* `enas_cnn_network.py`: the architecture of the searched network
* `enas_operations.py`: the copy of the [https://github.com/microsoft/nni/blob/master/examples/nas/enas/ops.py](https://github.com/microsoft/nni/blob/master/examples/nas/enas/ops.py)
* `train_enas_cnn_network.py`: the training file of the micro network of enas

eg.  
`data`: the file folder that save CIFAR-10 datasets  
`checkpoint`: the file folder that save the json files searched by enas, for example: epoch_0.json、...、epoch_149.json  
`checkpoint_pth`: the file folder that save the checkpoint file that after retraining 
