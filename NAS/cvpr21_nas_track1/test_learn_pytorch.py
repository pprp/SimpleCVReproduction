import torch 
import torch.nn as nn 

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # buffer = torch.randn(2, 3)  # tensor
        # self.register_buffer('my_buffer', buffer)

        # param = nn.Parameter(torch.randn(3, 3))  # 普通 Parameter 对象
        # self.register_parameter("param", param)

        self.param = nn.Parameter(torch.randn(2,4))


    def forward(self, x):
        # 可以通过 self.param 和 self.my_buffer 访问
        pass

model = MyModel()
for param in model.parameters():
    print(param)
print("----------------")
for buffer in model.buffers():
    print(buffer)
print("----------------")
print(model.state_dict())

a = nn.BatchNorm2d(5, affine=True, track_running_stats=True)
b = nn.BatchNorm2d(5, affine=True, track_running_stats=False)
c = nn.BatchNorm2d(5, affine=False, track_running_stats=True)
d = nn.BatchNorm2d(5, affine=False, track_running_stats=False)