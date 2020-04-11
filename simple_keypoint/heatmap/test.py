import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets
from model import KeyPointModel
from utils import *
from utils import compute_loss


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def flip_tensor(x):
  return torch.flip(x, [3])

if __name__ == "__main__":
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                             std=[0.2479, 0.2475, 0.2485])
    ])

    dataset = KeyPointDatasets(root_dir="./data", transforms=transforms_all)

    dataloader = DataLoader(dataset, shuffle=True,
                            batch_size=10, collate_fn=dataset.collect_fn)

    model = KeyPointModel()

    for iter, (image, label) in enumerate(dataloader):
        print(image.shape)
        heatmap = model(image)
        # hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        heatmap = torch.sigmoid(heatmap)
        heatmap = _nms(heatmap)

        batch, height, width = heatmap.size()
        print(heatmap.shape, heatmap[0])


