import h5py
import numpy as np
from nasbench import api
from tqdm import tqdm

NASBENCH_FULL_TFRECORD = "data/nasbench_full.tfrecord"
NASBENCH_HDF5 = "data/nasbench.hdf5"
LABEL2ID = {
    "input": -1,
    "output": -2,
    "conv3x3-bn-relu": 0,
    "conv1x1-bn-relu": 1,
    "maxpool3x3": 2
}

nasbench = api.NASBench(NASBENCH_FULL_TFRECORD)

metrics_ = []
operations_ = []
adjacency_ = []
trainable_parameters_ = []
hash_ = []
num_vertices_ = []

for hashval in tqdm(nasbench.hash_iterator()):
    metadata, metrics = nasbench.get_metrics_from_hash(hashval)
    hash_.append(hashval.encode())
    trainable_parameters_.append(metadata["trainable_parameters"])
    num_vertices = len(metadata["module_operations"])
    num_vertices_.append(num_vertices)
    assert num_vertices <= 7

    adjacency_padded = np.zeros((7, 7), dtype=np.int8)
    adjacency = np.array(metadata["module_adjacency"], dtype=np.int8)
    adjacency_padded[:adjacency.shape[0], :adjacency.shape[1]] = adjacency
    adjacency_.append(adjacency_padded)

    operations = np.array(list(map(lambda t: LABEL2ID[t], metadata["module_operations"])), dtype=np.int8)
    operations_padded = np.zeros((7, ), dtype=np.int8)
    operations_padded[:operations.shape[0]] = operations
    operations_.append(operations_padded)

    metrics_.append([])
    for epoch in [4, 12, 36, 108]:
        converted_metrics = []
        for seed in range(3):
            cur = metrics[epoch][seed]
            converted_metrics.append(np.array([[cur[t + "_training_time"],
                                                cur[t + "_train_accuracy"],
                                                cur[t + "_validation_accuracy"],
                                                cur[t + "_test_accuracy"]] for t in ["halfway", "final"]
                                               ], dtype=np.float32))
        metrics_[-1].append(converted_metrics)
hash_ = np.array(hash_)
operations_ = np.stack(operations_)
adjacency_ = np.stack(adjacency_)
trainable_parameters_ = np.array(trainable_parameters_, dtype=np.int32)
metrics_ = np.array(metrics_, dtype=np.float32)
num_vertices_ = np.array(num_vertices_, dtype=np.int8)

with h5py.File(NASBENCH_HDF5, "w") as fp:
    fp.create_dataset("hash", data=hash_)
    fp.create_dataset("num_vertices", data=num_vertices_)
    fp.create_dataset("trainable_parameters", data=trainable_parameters_)
    fp.create_dataset("adjacency", data=adjacency_)
    fp.create_dataset("operations", data=operations_)
    fp.create_dataset("metrics", data=metrics_)
