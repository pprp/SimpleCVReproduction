import torch
import math

from torch.utils.data.distributed import DistributedSampler


class OFADistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, sub_index_list=None):
        super(OFADistributedSampler, self).__init__(
            dataset, num_replicas, rank, shuffle)

        self.sub_index_list = sub_index_list

        self.num_samples = int(
            math.ceil(len(self.sub_index_list)*1.0/self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(
            len(self.sub_index_list), generator=g).tolist()

        indices += indices[:(self.total_size - len(indices))]
        indices = self.sub_index_list[indices].tolist()
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
