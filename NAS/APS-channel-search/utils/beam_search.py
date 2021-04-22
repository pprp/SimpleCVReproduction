""" Code modified from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding """

import operator
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from pdb import set_trace as br

MAX_LENGTH = 100000

class BeamSearchNode(object):
    def __init__(self, inputs, hiddenstate, hiddencell, previousNode, wordId, logProb, logits, entropy, length):
        self.i = inputs
        self.h = hiddenstate
        self.c = hiddencell
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.logits = logits
        self.entropy = entropy
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.logp < other.logp

def controller_step(controller, node):
    """ Forward one step of LSTM controller.
    Args:
      controller: the LSTM controller;
      node: the current node (with its input, cell and hidden, where:
          1) inputs: tensor of shape [1, embed_dim];
          2) cell: list of tensors of shape [1, hidden_dim], len=layers;
          3) hidden: list of tensors of shape [1, hidden_dim len=layers;

    Return:
      cell: ...
      hidden: ...
      logits: tensor of [vocab_size], the output
      logp: tensor of shape [vocab_size],logP of curretn step
      entropy: a scalar of entropy of current step
    """

    inputs, cell, hidden = node.i, node.c, node.h
    next_c, next_h = controller.stack_lstm(inputs, cell, hidden)
    logits = controller.w_soft(next_h[-1]).view(-1)
    if controller.temperature is not None:
        logits /= controller.temperature
    if controller.tanh_constant is not None:
        op_tanh = controller.tanh_constant / controller.op_tanh_reduce
        logits = op_tanh * controller.tanh(logits)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    entropy = - (log_probs * probs).sum()
    return next_c, next_h, logits, log_probs, entropy

def construct_input(controller, node, curr_action):
    # back-track to find historical archs
    device = controller.device
    action = torch.tensor(curr_action).to(device).view(1)
    archs = []
    archs.append(curr_action)

    # begin back trace
    while node.prevNode != None:
        node = node.prevNode
        if node.wordid == controller.n_ops:
            # controller.n_ops is SOS, so do not add the root node
            break
        archs.append(node.wordid)

    archs = torch.tensor(archs[::-1]).to(device)
    layer_idx = archs.size(0)
    assert layer_idx <= controller.num_search_layers, "archs depther than layers to search"
    flops_curr = controller._compute_flops(archs) # NOTE: blockwise will be considered in _compute_flops
    flops_left = ((controller.max_flops - flops_curr) / controller.max_flops).view((1, 1)).to(device)  # tensor
    layer_idx_onehot = torch.tensor(np.eye(controller.num_search_layers)[layer_idx].reshape(1, -1).astype(np.float32)).to(device)
    # concat [1, 80] with [1, 19] and [1,1] along dim 1
    inputs = torch.cat(tuple([controller.w_emb(action), layer_idx_onehot, flops_left]), dim=-1)
    return inputs

def beam_decode(controller, topk):
    '''
    Args:
      deocder_input: the init input tensor [1, input_dim]
      decoder_cell: a list of tensors with length equal to layers of LSTM, each of shape [1, HC], where HC is cell dimension
      decoder_hidden:: a list of tensors with length equal to layers of LSTM, each of shape [1, H], where H is hidden dimension
      beam_width: an int
      topk: an int, the number of sequence to generate
      controller: the LSTM controller

    Return:
      decoded_batch: a list of torch.tensor of shape [controller.num_search_layers]
      decoded_logits: list of list of tensors of shape [controller.n_ops], first list: topk; second list: controller.num_search_layers
      decoded_logP: a tesnor of accumulated logP, [topk]
      decoded_entropy: a tensor of accumulated entropy: [topk]
    '''

    # prepare params for beam search
    beam_width = controller.n_ops
    num_arch = controller.num_search_layers
    SOS_token = controller.n_ops
    device = controller.device

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # init starting point
    n_word_index = SOS_token # start of the sequence
    decoder_input, decoder_cell, decoder_hidden = controller._init_nodes()

    # starting node -  layerwise hidden vector, previous node, word id, logp, logits, entropy, length
    node = BeamSearchNode(decoder_input, decoder_hidden, decoder_cell, previousNode=None, \
        wordId=n_word_index, logProb=0, logits=[], entropy=0, length=1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > MAX_LENGTH: break

        # fetch the best node and its things
        score, n = nodes.get()
        decoder_hidden = n.h
        decoder_cell = n.c

        # NOTE: stop when the tree has depth = num layers + 1 (SOS)
        if n.leng == (num_arch + 1):
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_cell, decoder_hidden, decoder_output, log_probs, entropy = controller_step(controller, n)

        # PUT HERE REAL BEAM SEARCH OF TOP
        # log_prob and indexes: [1, beam_width]
        log_prob, indexes = torch.topk(log_probs, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[new_k].item()
            log_p = log_prob[new_k].item()
            decoder_input = construct_input(controller, n, decoded_t)
            node = BeamSearchNode(decoder_input, decoder_hidden, decoder_cell, n, decoded_t, n.logp+log_p, \
                n.logits.copy()+[decoder_output.squeeze()], n.entropy+entropy, n.leng+1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))

        # increase qsize
        qsize += len(nextnodes) - 1

    # beam search done. Take the sequences out
    decoded_batch = []
    decoded_logits = []
    decoded_logP = []
    decoded_entropy = []

    for score, n in sorted(endnodes, key=operator.itemgetter(0)):

        decoded_logP.append(n.logp)
        decoded_entropy.append(n.entropy)
        decoded_logits.append(n.logits) # logits list is in correct order

        utterance = []
        utterance.append(n.wordid)

        # begin back trace
        while n.prevNode != None:
            n = n.prevNode
            if n.wordid == SOS_token:
                # do not add the root node
                break
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        decoded_batch.append(utterance)

    decoded_batch = torch.tensor(decoded_batch).to(device)
    decoded_entropy = torch.tensor(decoded_entropy).to(device)
    decoded_logP = torch.tensor(decoded_logP).to(device)
    return decoded_batch, decoded_logits, decoded_logP, decoded_entropy
