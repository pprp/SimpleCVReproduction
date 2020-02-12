import torch
import torchvision
'''
# 主要介绍张量操作
'''
### 张量形变
# 相比torch.view，
# torch.reshape可以自动处理输入张量不连续的情况。
tensor = torch.rand(2, 3, 4)
shape = (6, 4)
tensor = torch.reshape(tensor, shape)

### 打乱顺序
# 打乱第一个维度
tensor = tensor[torch.randperm(tensor.size(0))]

### 水平翻转
# tensor [n, c, h, w]
tensor = tensor[:, :, :, torch.arange(tensor.size(3) - 1, -1, -1).long()]

### 复制
# numpy.copy() tensor.clone()
# Operation                 |  New/Shared memory | Still in computation graph |
tensor.clone()  # |        New         |          Yes               |
tensor.detach()  # |      Shared        |          No                |
tensor.detach().clone()  # |        New         |          No                |

### 拼接
'''
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，
而torch.stack会新增一维。
例如当参数是3个10x5的张量，
torch.cat的结果是30x5的张量，
而torch.stack的结果是3x10x5的张量。
'''
tensor = torch.cat(list_of_tensors, dim=0)
tensor = torch.stack(list_of_tensors, dim=0)

### one-hot
tensor = torch.tensor([0, 2, 1, 3])
N = tensor.size(0)
num_classes = 4

one_hot = torch.zeros(N, num_classes).long()
one_hot.scatter_(dim=1,
            index=torch.unsqueeze(tensor,dim=1),
            src=torch.ones(N, num_classes).long()
            )

### 得到非零元素
torch.nonzero(tensor)  # index of non-zero elements
torch.nonzero(tensor == 0)  # index of zero elements
torch.nonzero(tensor).size(0)  # number of non-zero elements
torch.nonzero(tensor == 0).size(0)  # number of zero elements

### 判断两个张量相等
torch.allclose(tensor1, tensor2)  # float tensor
torch.equal(tensor1, tensor2)  # int tensor

### 张量扩展
# Expand tensor of shape 64*512 to shape 64*512*7*7.
tensor = torch.rand(64,512)
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)

### 矩阵乘法
# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).
result = torch.mm(tensor1, tensor2)

# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)
result = torch.bmm(tensor1, tensor2)

# Element-wise multiplication.
result = tensor1 * tensor2

### 计算两组数据之间的两两欧式距离
# 利用broadcast机制
dist = torch.sqrt(torch.sum((X1[:, None, :] - X2)**2, dim=2))
