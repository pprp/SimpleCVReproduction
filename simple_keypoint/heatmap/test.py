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

    for iter, (image, label) in enumerate(dataloader):
        print(image.shape)
        bs = image.shape[0]
        hm = model(image)

        hm = _nms(hm)

        hm = hm.detach().numpy()

        for i in range(bs):
            hm = hm[i]
            hm = np.maximum(hm, 0)
            hm = hm/np.max(hm)
            hm = normalization(hm)
            hm = np.uint8(255 * hm)
            # heatmap = torch.sigmoid(heatmap)

            hm = cv2.cvtColor(hm,cv2.COLOR_RGB2BGR)

            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)

            print(hm.shape)

            cv2.imwrite("output_%d_%d.jpg" % (iter, i), hm)
            cv2.waitKey(0)
            # batch, height, width = heatmap.size()
            # print(heatmap.shape, heatmap[0])
