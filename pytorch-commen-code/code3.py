# 转换numpy tensor PIL opencv
import torch
import torchvision
import PIL
import cv2
import numpy as np

tensor = torch.zeros((32, 3, 8, 16))

# torch :  [bs, c, h ,w] 并且在0-1需要进行转置和规范化
#       :  shape or size()
# PIL   :  [w, h] 通道是顺序RGB,save的时候直接RGB
#       :  size
# cv2   :  [] 
#       :  通道顺序BGR,save的时候先转BGR2RGB在保存
#       :  cv2中存储是numpy

# tensor -> numpy
ndarray = tensor.cpu().numpy()

# numpy -> tensor
tensor = torch.from_numpy(ndarray).float()
tensor = torch.from_numpy(ndarray.copy()).float()


# tensor -> PIL
tensor = tensor[0] # 取其中一张图片
image = PIL.Image.fromarray(
        torch.clamp(tensor * 255, min=0, max=255)
        .byte()
        .permute(1,2,0)
        .cpu()
        .numpy())

print(image.size)
# or
image = torchvision.transforms.functional.to_pil_image(tensor)
# or
topil = torchvision.transforms.ToPILImage()
image = topil(tensor)

# PIL -> tensor
path = "./test.jpg"
tensor = torch.from_numpy(
         np.asarray(PIL.Image.open(path))
         .permute(2,0,1)
         .float() / 255
         )
# or
tensor = torchvision.transforms.functional
         .to_tensor(PIL.Image.open(path))

# numpy -> PIL
image = PIL.Image.fromarray(ndarray.astype(np.uint8))

# PIL -> numpy
ndarray = np.asarray(PIL.Image.open(path))

# cv2 -> PIL
cv2_img = cv2.imread("./test.jpg")
pil_img = PIL.Image.fromarray(
          cv2.cvtColor(cv2_img,cv2.COLOR_BGR2RGB)
          )

# PIL -> cv2
pil_img = PIL.Image.open("./test.jpg")
cv2_img = cv2.cvtColor(numpy.asarray(pil_img),
            cv2.COLOR_RGB2BGR) 
