
import torchvision

from receptivefield import receptivefield


if __name__ == '__main__':
  # get standard VGG
  net = torchvision.models.vgg16()

  # change the forward function to output convolutional features only.
  # otherwise the output is fully-connected and the receptive field is the whole image.
  def features_only(self, x):
    return self.features(x)
  net.forward = features_only.__get__(net)  # bind method

  # compute receptive field for this input shape
  rf = receptivefield(net, (1, 3, 416, 416))

  # print to console, and visualize
  print(rf)
  rf.show()
