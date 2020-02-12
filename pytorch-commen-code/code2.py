import torch

tensor = torch.randn(3,4,5)
print(tensor.type())
print(tensor.size())
# size不是shape
print(tensor.dim())
print(tensor.shape)

# 骚操作：命名张量
images = torch.randn(3,3,4,4)
print(images.sum(dim=1).shape)
print(images.select(dim=1, index=0).shape)

# pytorch1.3之后
NCHW = ["N", "C, "H", "W"]
images = torch.randn(3,3,4,4,names=NCHW)
images.sum("C")
images.select('C', index=0)

# 将张量尺寸与指定顺序对齐。
tensor = tensor.align_to('N', 'C', 'H', 'W')
# won't use it in the future.

# data type transfer
torch.set_default_tensor_type(torch.FloatTensor)
tensor = tensor.cuda()
tensor = tensor.cpu()

tensor = tensor.float()
tensor = tensor.long()

