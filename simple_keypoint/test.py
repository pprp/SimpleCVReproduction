import glob
import cv2
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets
from model import KeyPointModel
import PIL

SIZE = 480, 360

transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 360)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                         std=[0.2479, 0.2475, 0.2485])
])

datasets_test = KeyPointDatasets(root_dir="./data", transforms=transforms_test)


dataloader_test = DataLoader(
    datasets_test, batch_size=4, shuffle=True, collate_fn=datasets_test.collect_fn)

model = KeyPointModel()

model.load_state_dict(torch.load("weights/epoch_490_0.000.pt"))

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

output = model(img_tensor_list)

print(output.shape)

bs = img_tensor_list.shape[0]

for i in range(bs):
    img_path = img_list[i]
    img = cv2.imread(img_path)

    point_ratio = output[i]

    # print(point_ratio.shape)

    x, y = SIZE[0] * point_ratio[0], SIZE[1] * point_ratio[1]

    x = int(x.item())
    y = int(y.item())

    # print(x, y)

    cv2.circle(img, (x, y), 5, (255, 0, 0), thickness=-1)

    print("./output/%s_out.jpg" % (img_name_list[i]))

    cv2.imwrite("./output/%s_out.jpg" % (img_name_list[i]), img)

