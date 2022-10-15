
import torch.nn as nn

from receptivefield import receptivefield


if __name__ == '__main__':
  # define a network with different kinds of standard layers
  net = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
    nn.BatchNorm2d(num_features=32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3),
  )

  # compute receptive field for this input shape
  rf = receptivefield(net, (1, 3, 32, 48))

  # print to console, and visualize
  print(rf)
  rf.show()
