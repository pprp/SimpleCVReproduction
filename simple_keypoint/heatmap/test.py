import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets
from model import KeyPointModel
from utils import *
from utils import compute_loss

parser = argparse.ArgumentParser(description="model path")
parser.add_argument('--model', type=str, default="")

args = parser.parse_args()


def _gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=1):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def flip_tensor(x):
    return torch.flip(x, [3])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


if __name__ == "__main__":
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                             std=[0.2479, 0.2475, 0.2485])
    ])

    dataset = KeyPointDatasets(root_dir="./data", transforms=transforms_all)

    dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=1,
                            collate_fn=dataset.collect_fn)

    model = KeyPointModel()
    model.load_state_dict(torch.load(args.model))

    for iter, (image, label) in enumerate(dataloader):
        # print(image.shape)
        bs = image.shape[0]
        hm = model(image)

        hm = _nms(hm)

        scores, inds, clses, ys, xs = _topk(hm,K=1)

        print(scores, '\n', inds, '\n', clses, '\n', ys, '\n', xs)

        hm = hm.detach().numpy()

        for i in range(bs):
            hm = hm[i]
            hm = np.maximum(hm, 0)
            hm = hm/np.max(hm)
            hm = normalization(hm)
            hm = np.uint8(255 * hm)
            hm = hm[0]
            # heatmap = torch.sigmoid(heatmap)

            # hm = cv2.cvtColor(hm, cv2.COLOR_RGB2BGR)

            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

            print(hm.shape)

            cv2.imwrite("./test_output/output_%d_%d.jpg" % (iter, i), hm)
            cv2.waitKey(0)
            # batch, height, width = heatmap.size()
            # print(heatmap.shape, heatmap[0])
