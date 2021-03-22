import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
# torch.set_default_tensor_type(torch.DoubleTensor)


class Nb101Dataset(Dataset):
    MEAN = 0.908192
    STD = 0.023961

    def __init__(self, split=None, debug=False):
        self.hash2id = dict()
        with h5py.File("data/nasbench.hdf5", mode="r") as f:
            for i, h in enumerate(f["hash"][()]):
                self.hash2id[h.decode()] = i
            self.num_vertices = f["num_vertices"][()]
            self.trainable_parameters = f["trainable_parameters"][()]
            self.adjacency = f["adjacency"][()]
            self.operations = f["operations"][()]
            self.metrics = f["metrics"][()]
        self.random_state = np.random.RandomState(0)
        if split is not None and split != "all":
            self.sample_range = np.load("data/train.npz")[str(split)]
        else:
            self.sample_range = list(range(len(self.hash2id)))
        self.debug = debug
        self.seed = 0

    def __len__(self):
        return len(self.sample_range)

    def _check(self, item):
        n = item["num_vertices"]
        ops = item["operations"]
        adjacency = item["adjacency"]
        mask = item["mask"]
        assert np.sum(adjacency) - np.sum(adjacency[:n, :n]) == 0
        assert np.sum(ops) == n
        assert np.sum(ops) - np.sum(ops[:n]) == 0
        assert np.sum(mask) == n and np.sum(mask) - np.sum(mask[:n]) == 0

    def mean_acc(self):
        return np.mean(self.metrics[:, -1, self.seed, -1, 2])

    def std_acc(self):
        return np.std(self.metrics[:, -1, self.seed, -1, 2])

    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN

    def resample_acc(self, index, split="val"):
        # when val_acc or test_acc are out of range
        assert split in ["val", "test"]
        split = 2 if split == "val" else 3
        for seed in range(3):
            acc = self.metrics[index, -1, seed, -1, split]
            if not self._is_acc_blow(acc):
                return acc
        if self.debug:
            print(index, self.metrics[index, -1, :, -1])
            raise ValueError
        return np.array(self.MEAN)

    def _is_acc_blow(self, acc):
        return acc < 0.2

    def __getitem__(self, index):
        index = self.sample_range[index]
        val_acc, test_acc = self.metrics[index, -1, self.seed, -1, 2:]

        if self._is_acc_blow(val_acc):
            val_acc = self.resample_acc(index, "val")

        if self._is_acc_blow(test_acc):
            test_acc = self.resample_acc(index, "test")

        n = self.num_vertices[index]

        ops_onehot = np.array([[i == k + 2 for i in range(5)]
                               for k in self.operations[index]], dtype=np.float32)

        if n < 7:
            ops_onehot[n:] = 0.

        result = {
            "num_vertices": n,
            "adjacency": self.adjacency[index],
            "operations": ops_onehot,
            "mask": np.array([i < n for i in range(7)]).astype(np.float32),
            "val_acc": self.normalize(val_acc).astype(np.float32),
            "test_acc": self.normalize(test_acc).astype(np.float32)
        }
        if self.debug:
            self._check(result)
        return result
