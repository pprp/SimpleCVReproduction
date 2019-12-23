import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F

BLANK_LABEL = 0

def get_dataset_path():
    """ Set dataset path
    """
    file_path = os.path.realpath(__file__)
    cur_dir = os.path.split(file_path)[0]
    return os.path.join(cur_dir, 'data')

DATASET_PATH = get_dataset_path()

if not os.path.exists(DATASET_PATH):
    os.mkdir(DATASET_PATH)

def _is_tuple(tuple_like):
    return isinstance(tuple_like, tuple)

def tensor_to_variable(tensor, volatile):
    if _is_tuple(tensor):
        return tuple((tensor_to_variable(x, volatile) for x in tensor))
    else:
        return Variable(tensor, volatile)

def to_gpu(item):
    if _is_tuple(item):
        return tuple((to_gpu(x) for x in item))
    else:
        return item.cuda()

def get_prediction(output, prob=True, lengths=None):
    """ Get prediction for sequence classification.

       :param output(torch.FloatTensor): output of model: seq_len x batch_size x channels
       :param prob(boolean): use `softmax` to conver `output` to probability or not
       :param length(list or None): lengths of batch elements. if have the same length, it can
                                    be inferred from `output`.
    """
    seq_len, batch_size, channels = output.size()
    if lengths is None:
        lengths = [seq_len] * batch_size
    else:
        assert len(lengths) == batch_size
    output = output.view((-1, channels))

    if isinstance(output, Variable):
        output = output.data
    if prob:
        output = Variable(output, volatile=True)
        output = F.softmax(output).data
    if output.is_cuda:
        output = output.cpu()

    prob, max_ind = torch.max(output, dim=1)
    prob = prob.view((seq_len, batch_size))
    max_ind = max_ind.view((seq_len, batch_size))
    pred_probs = []
    pred_labels = []
    for b in range(batch_size):
        bprob = prob[:, b]
        bind = max_ind[:, b]
        prev_ind = BLANK_LABEL
        pred_label, pred_prob = [], []
        for t in range(lengths[b]):
            if bind[t] != prev_ind and bind[t] != BLANK_LABEL:
                pred_label.append(bind[t])
                pred_prob.append(bprob[t])
            prev_ind = bind[t]
        pred_probs.append(pred_prob)
        pred_labels.append(pred_label)
    return pred_labels, pred_probs

    
