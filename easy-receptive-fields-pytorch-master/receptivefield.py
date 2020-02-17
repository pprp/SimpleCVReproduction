
from collections import namedtuple
import math
import torch as t
import torch.nn as nn


Size = namedtuple('Size', ('w', 'h'))
Vector = namedtuple('Vector', ('x', 'y'))

class ReceptiveField(namedtuple('ReceptiveField', ('offset', 'stride', 'rfsize', 'outputsize', 'inputsize'))):
  """Contains information of a network's receptive fields (RF).
  The RF size, stride and offset can be accessed directly,
  or used to calculate the coordinates of RF rectangles using
  the convenience methods."""

  def left(self):
    """Return left (x) coordinates of the receptive fields."""
    return t.arange(float(self.outputsize.w)) * self.stride.x + self.offset.x
    
  def top(self):
    """Return top (y) coordinates of the receptive fields."""
    return t.arange(float(self.outputsize.h)) * self.stride.y + self.offset.y
  
  def hcenter(self):
    """Return center (x) coordinates of the receptive fields."""
    return self.left() + self.rfsize.w / 2
    
  def vcenter(self):
    """Return center (y) coordinates of the receptive fields."""
    return self.top() + self.rfsize.h / 2
    
  def right(self):
    """Return right (x) coordinates of the receptive fields."""
    return self.left() + self.rfsize.w

  def bottom(self):
    """Return bottom (y) coordinates of the receptive fields."""
    return self.top() + self.rfsize.h
  
  def rects(self):
    """Return a list of rectangles representing the receptive fields of all output elements. Each rectangle is a tuple (x, y, width, height)."""
    return [(x, y, self.rfsize.w, self.rfsize.h) for x in self.left().numpy() for y in self.top().numpy()]
  

  def show(self, image=None, axes=None, show=True):
    """Visualize receptive fields using MatPlotLib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if image is None:
      # create a checkerboard image for the background
      xs = t.arange(self.inputsize.w).unsqueeze(1)
      ys = t.arange(self.inputsize.h).unsqueeze(0)
      image = (xs.remainder(8) >= 4) ^ (ys.remainder(8) >= 4)
      image = image * 128 + 64

    if axes is None:
      (fig, axes) = plt.subplots(1)

    # convert image to numpy and show it
    if isinstance(image, t.Tensor):
      image = image.numpy().transpose(-1, -2)
    axes.imshow(image, cmap='gray', vmin=0, vmax=255)

    rect_density = self.stride.x * self.stride.y / (self.rfsize.w * self.rfsize.h)
    rects = self.rects()

    for (index, (x, y, w, h)) in enumerate(rects):  # iterate RFs
      # show center marker
      marker, = axes.plot(x + w/2, y + w/2, marker='x')

      # show rectangle with some probability, since it's too dense.
      # also, always show the first and last rectangles for reference.
      if index == 0 or index == len(rects) - 1 or t.rand(1).item() < rect_density:
        axes.add_patch(patches.Rectangle((x, y), w, h, facecolor=marker.get_color(), edgecolor='none', alpha=0.5))
        first = False
    
    # set axis limits correctly
    axes.set_xlim(self.left().min().item(), self.right().max().item())
    axes.set_ylim(self.top().min().item(), self.bottom().max().item())
    axes.invert_yaxis()

    if show: plt.show()


(x_dim, y_dim) = (-1, -2)  # indexes of spatial dimensions in tensors


def receptivefield(net, input_shape, device='cpu'):
  """Computes the receptive fields for the given network (nn.Module) and input shape, given as a tuple (images, channels, height, width).
  Returns a ReceptiveField object."""

  if len(input_shape) < 4:
    raise ValueError('Input shape must be at least 4-dimensional (N x C x H x W).')

  # make gradients of some problematic layers pass-through
  hooks = []
  def insert_hook(module):
    if isinstance(module, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d)):
      hook = _passthrough_grad
      if isinstance(module, nn.MaxPool2d):
        hook = _maxpool_passthrough_grad
      hooks.append(module.register_backward_hook(hook))
  net.apply(insert_hook)

  # remember whether the network was in train/eval mode and set to eval
  mode = net.training
  net.eval()

  # compute forward pass to prepare for gradient computation
  input = t.ones(input_shape, requires_grad=True, device=device)
  output = net(input)

  if output.dim() < 4:
    raise ValueError('Network is fully connected (output should have at least 4 dimensions: N x C x H x W).')
  
  # output feature map size
  outputsize = Size(output.shape[x_dim], output.shape[y_dim])
  if outputsize.w < 2 and outputsize.h < 2:  # note: no error if only one dim is singleton
    raise ValueError('Network output is too small along spatial dimensions (fully connected).')

  # get receptive field bounding box, to compute its size.
  # the position of the one-hot output gradient (pos) is stored for later.
  (x1, x2, y1, y2, pos) = _project_rf(input, output, return_pos=True)
  rfsize = Size(x2 - x1 + 1, y2 - y1 + 1)

  # do projection again with one-cell offsets, to calculate stride
  (x1o, _, _, _) = _project_rf(input, output, offset_x=1)
  (_, _, y1o, _) = _project_rf(input, output, offset_y=1)
  stride = Vector(x1o - x1, y1o - y1)

  if stride.x == 0 and stride.y == 0:  # note: no error if only one dim is singleton
    raise ValueError('Input tensor is too small relative to network receptive field.')

  # compute offset between the top-left corner of the receptive field in the
  # actual input (x1, y1), and the top-left corner obtained by extrapolating
  # just based on the output position and stride (the negative terms below).
  offset = Vector(x1 - pos[x_dim] * stride.x, y1 - pos[y_dim] * stride.y)

  # remove the hooks from the network, and restore training mode
  for hook in hooks: hook.remove()
  net.train(mode)

  # return results in a nicely packed structure
  inputsize = Size(input_shape[x_dim], input_shape[y_dim])
  return ReceptiveField(offset, stride, rfsize, outputsize, inputsize)


def _project_rf(input, output, offset_x=0, offset_y=0, return_pos=False):
  """Project one-hot output gradient, using back-propagation, and return its bounding box at the input."""

  # create one-hot output gradient tensor, with 1 in the center (spatially)
  pos = [0] * len(output.shape)  # index 0th batch/channel/etc
  pos[x_dim] = math.ceil(output.shape[x_dim] / 2) - 1 + offset_x
  pos[y_dim] = math.ceil(output.shape[y_dim] / 2) - 1 + offset_y

  out_grad = t.zeros(output.shape)
  out_grad[tuple(pos)] = 1

  # clear gradient first
  if input.grad is not None:
    input.grad.zero_()

  # propagate gradient of one-hot cell to input tensor
  output.backward(gradient=out_grad, retain_graph=True)

  # keep only the spatial dimensions of the gradient at the input, and binarize
  in_grad = input.grad[0, 0]
  is_inside_rf = (in_grad != 0.0)

  # x and y coordinates of where input gradients are non-zero (i.e., in the receptive field)
  xs = is_inside_rf.any(dim=y_dim).nonzero()
  ys = is_inside_rf.any(dim=x_dim).nonzero()

  if xs.numel() == 0 or ys.numel() == 0:
    raise ValueError('Could not propagate gradient through network to determine receptive field.')

  # return bounds of receptive field
  bounds = (xs.min().item(), xs.max().item(), ys.min().item(), ys.max().item())
  if return_pos:  # optionally, also return position of one-hot output gradient
    return (*bounds, pos)
  return bounds


def _passthrough_grad(self, grad_input, grad_output):
  """Hook to bypass normal gradient computation (of first input only)."""
  if isinstance(grad_input, tuple) and len(grad_input) > 1:
    # replace first input's gradient only
    return (grad_output[0], *grad_input[1:])
  else:  # single input
    return grad_output

def _maxpool_passthrough_grad(self, grad_input, grad_output):
  """Hook to bypass normal gradient computation of nn.MaxPool2d."""
  assert isinstance(self, nn.MaxPool2d)
  if self.dilation != 1 and self.dilation != (1, 1):
    raise ValueError('Dilation != 1 in max pooling not supported.')

  # backprop through a nn.AvgPool2d with same args as nn.MaxPool2d
  with t.enable_grad():                               
    input = t.ones(grad_input[0].shape, requires_grad=True)
    output = nn.functional.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode)
    return t.autograd.grad(output, input, grad_output[0])


def run_test():
  """Tests various combinations of inputs and checks that they are correct."""
  # this is easy to do for convolutions since the RF is known in closed form.
  for kw in [1, 2, 3, 5]:  # kernel width
    for sx in [1, 2, 3]:  # stride in x
      for px in [1, 2, 3, 5]:  # padding in x
        (kh, sy, py) = (kw + 1, sx + 1, px + 1)  # kernel/stride/pad in y
        for width in range(kw + sx * 2, kw + 3 * sx + 1):  # enough width
          for height in range(width + 1, width + sy + 1):
            # create convolution and compute its RF
            net = nn.Conv2d(3, 2, (kh, kw), (sy, sx), (py, px))
            rf = receptivefield(net, (1, 3, height, width))

            print('Checking: ', rf)
            assert rf.rfsize.w == kw and rf.rfsize.h == kh
            assert rf.stride.x == sx and rf.stride.y == sy
            assert rf.offset.x == -px and rf.offset.y == -py
  print('Done, all tests passed.')


if __name__ == '__main__':
  run_test()

