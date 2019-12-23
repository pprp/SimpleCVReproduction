import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

class StackedRNN(nn.Module):
    """Stacked RNN
    """
    def __init__(self, input_size, output_size, hidden_size, number_layers):
        super(StackedRNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, number_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.nlayers = number_layers
        self.nhid = hidden_size

    def init_hidden(self, bsz, volatile=False):
        weight = next(self.parameters()).data
    
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_(), volatile=volatile),
                Variable(weight.new(self.nlayers, bsz, self.nhid).zero_(), volatile=volatile))

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        # output is seq_len x batch_size x hidden_size
        seq_len = output.size()[0]
        output = torch.stack([self.fc(output[t]) for t in range(seq_len)])

        return output, hidden

