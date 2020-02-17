
import torchvision

from receptivefield import receptivefield


if __name__ == '__main__':
  # get standard ResNet
  net = torchvision.models.resnet18()

  # ResNet block to compute receptive field for
  block = 2
  
  # change the forward function to output convolutional features only.
  # otherwise the output is fully-connected and the receptive field is the whole image.
  def features_only(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    if block == 0: return x
    
    x = self.layer1(x)
    if block == 1: return x
    
    x = self.layer2(x)
    if block == 2: return x
    
    x = self.layer3(x)
    if block == 3: return x
    
    x = self.layer4(x)
    return x
  net.forward = features_only.__get__(net)  # bind method
  

  # compute receptive field for this input shape
  rf = receptivefield(net, (1, 3, 224, 224))

  # print to console, and visualize
  print(rf)
  rf.show()
