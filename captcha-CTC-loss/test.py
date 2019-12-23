import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from model import StackedRNN
use_cuda = torch.cuda.is_available()

input_size, output_size = 180, 11
hidden_size = 512
number_layer = 2
## get model
def get_model():
    return StackedRNN(input_size, output_size, hidden_size, number_layer)

def load_model(model_path):
    if not os.path.exists(model_path):
        raise RuntimeError('cannot find model path: {}'.format(model_path))
    checkpoint = torch.load(model_path)
    print 'load model done.'
    print 'accuracy: {:.2f}'.format(checkpoint['accuracy'])
    return checkpoint['state_dict']

normalized = True
if normalized:
    from dataset import mean, std
else:
    mean = [0. for _ in range(3)]
    std = [1. for _ in range(3)]

def read_preprocess_image(img_path):
    if not os.path.exists(img_path):
        raise RuntimeError('cannot find image: {}'.format(img_path))
    im = cv2.imread(img_path).astype(np.float32)
    im /= 255
    im -= mean
    im /= std
    # im is HxWxC now
    # change to Wx1x(HxC)
    h, w, c = im.shape
    im = torch.from_numpy(im).float()
    im = im.permute(1, 0, 2).contiguous().view((w, -1)).unsqueeze(1)
    return im

def demo(img_path, model_path):
    model = get_model()
    if use_cuda:
        model = model.cuda()
    model.load_state_dict(load_model(model_path))
    input = read_preprocess_image(img_path)
    if use_cuda:
        input = input.cuda()
    input = Variable(input, volatile=True)
    hidden = model.init_hidden(1, volatile=True)

    out, _ = model(input, hidden)
    _, max_id = out.data.squeeze().max(dim=1)

    ret_labels = decode(max_id)

    ret_labels = [(x[0]-1, x[1], x[2]) for x in ret_labels]

    vis(img_path, ret_labels)

    probs = F.softmax(out.view((-1, out.size(2)))).view(out.size())
    show_prob(probs.data, [x[0] for x in ret_labels])

    plt.show()

def show_prob(probs, labels=None):
    if labels is None:
        labels = range(11)
    labels = sorted(list(set(labels)))
    num_subplots = len(labels)
    probs = probs.squeeze().transpose(1, 0)
    figure = plt.figure()
    for ind, label in enumerate(labels):
        prob = probs[label+1]        
        plt.subplot(num_subplots, 1, ind+1)
        plt.plot(list(prob))
        plt.title('prob of {:2d} vs time step'.format(label))
    

def vis(img_path, result):
    """ visualization
    """
    im = cv2.imread(img_path)
    h, w, c = im.shape
    for i, x in enumerate(result):
        cv2.line(im, (x[1], 0), (x[1], h), (255, 0, 0), 1)
        cv2.line(im, (x[2]-1, 0), (x[2]-1, h), (0, 0, 255), 1)
    label = [str(x[0]) for x in result]
    im = im[:,:,::-1]
    plt.imshow(im)
    plt.title(''.join(label))

def decode(raw_label_seq):
    """ Decode the raw label sequence
    """
    BLANK = 0
    if isinstance(raw_label_seq, list):
        raw_label_seq = torch.IntTensor(raw_label_seq)

    label_seq = list(raw_label_seq.squeeze()) + [BLANK]
    
    prev = BLANK

    length = len(label_seq)
    i = 0
    ret_labels = []
    while i < length:
        if label_seq[i] != prev:
            # period starts or ends
            if prev == BLANK:
                # start a new period
                start = i
            else:
                # end of a period
                ret_labels.append((prev, start, i))
                start = i
        prev = label_seq[i]
        i += 1
    return ret_labels

def help():
    print("""Usage:
test image_path trained_model_path""")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        help()
        sys.exit(-1)
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    demo(img_path, model_path)
