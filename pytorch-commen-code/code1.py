"""
http://bbs.cvmart.net/topics/1472?from=timeline&isappinstalled=0
来自极市平台@Jack Stark
"""

import torch
import torch.nn as nn
import torchvision

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.get_device_name(0))

# 复现性
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudann.backends = False

# 显卡设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

torch.cuda.empty_cache()#清空显存
'''
nvidia-smi --gpu-reset -i [gpu_id]
'''


