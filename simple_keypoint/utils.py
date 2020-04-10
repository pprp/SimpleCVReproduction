import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 360)),
    transforms.ToTensor(),
])

kp_datasets = KeyPointDatasets(root_dir="./data", transforms=trans)

data_loader = DataLoader(kp_datasets, num_workers=0, batch_size=4, shuffle=True,
                         collate_fn=kp_datasets.collect_fn
                         )

print("computing mean and std ....")

mean = 0.
std = 0.
n_sample = 0

for data, label in data_loader:
    bs = data.size(0)
    data = data.view(bs, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    n_sample += bs

mean /= n_sample
std /= n_sample

print("mean: {}\nstd: {}".format(mean, std))
