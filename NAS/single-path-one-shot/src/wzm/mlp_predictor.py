
import torch
import torch.nn as nn
import os
import numpy
import torch.nn.functional as F

class AccuracyPredictor(nn.Module):

    def __init__(self, hidden_size=512, input_dim=160, n_layers=3):
        super(AccuracyPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dim=input_dim
        layers = []
        for i in range(self.n_layers):
            layers.append(nn.Sequential(
                nn.Linear(self.input_dim if i == 0 else self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.1)))
        layers.append(nn.Linear(self.hidden_size, 1, bias=False))
        self.layers = nn.Sequential(*layers)


    def predict(self, arch):
        # score = self.layers(arch).squeeze(-1)
        arch=torch.tensor(arch, dtype=torch.float)
        score = self.layers(arch)
        return score

if __name__ == '__main__':
    net = AccuracyPredictor()
    score_list=[]
    path = '/home/wzm/test/vgg_nas/predict_checkpoint/{}.pth'.format(2)
    print('load predictor checkpoint from {}:'.format(path))
    net.load_state_dict(torch.load(path))
    arch_set=[[1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]]
    net.eval()


    for l in arch_set:
        score_list.append(net.predict(l))

    print('score list:',score_list)













