import glob
import os

import cv2
import numpy as np
import PIL
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets
from model import KeyPointModel

SIZE = 360, 480

transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                         std=[0.2479, 0.2475, 0.2485])
])

datasets_test = KeyPointDatasets(root_dir="./data", transforms=transforms_test)


dataloader_test = DataLoader(
    datasets_test, batch_size=4, shuffle=True, collate_fn=datasets_test.collect_fn)

model = KeyPointModel()
model.eval()

model.load_state_dict(torch.load("weights/epoch_90_0.000.pt"))

img_list = glob.glob(os.path.join("./data/images", "*.jpg"))

save_path = "./output"

img_tensor_list = []
img_name_list = []

for i in range(len(img_list)):
    img_path = img_list[i]
    img_name = os.path.basename(img_path)
    img_name_list.append(img_name)

    img = cv2.imread(img_path)
    img_tensor = transforms_test(img)
    img_tensor_list.append(img_tensor)

img_tensor_list = torch.stack(img_tensor_list, 0)

print(img_tensor_list.shape)

# part of it
# img_tensor_list = img_tensor_list[-10:]
# img_name_list = img_name_list[-10:]

heatmap = model(img_tensor_list)

# heatmap = torch.sigmoid(heatmap)

bs = img_tensor_list.shape[0]


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

for i in range(bs):
    # img_path = img_list[i]
    # img = cv2.imread(img_path)
    # img = cv2.resize(img, (480, 360))

    hm = heatmap[i].detach().numpy()

    hm = np.maximum(hm, 0)
    hm = hm/np.max(hm)
    hm = normalization(hm)

    hm = np.uint8(255 * hm)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm = cv2.resize(hm, (480, 360))

    superimposed_img = hm #* 0.4 + img

    # x, y = SIZE[0] * point_ratio[0], SIZE[1] * point_ratio[1]

    # x = int(x.item())
    # y = int(y.item())

    # cv2.circle(img, (x, y), 5, (0, 0, 255), thickness=-1)

    print("./output/%s_out.jpg" % (img_name_list[i]))

    cv2.imwrite("./output/%s_out.jpg" % (img_name_list[i]), superimposed_img)
