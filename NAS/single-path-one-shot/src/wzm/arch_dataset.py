import torch
import json
import pandas as pd
import numpy as np


def load_from_json(json_path):
    with open(json_path, "r") as fin:
        a = json.load(fin)
    res=[]
    for key in a.keys():
        sample=[]
        arch_set=a[key]['arch']
        score=a[key]['acc']
        for arch in arch_set:
            sample+=arch[1:]
        sample+=[score]
        res.append(sample)
    return res



class RegDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        super(RegDataset, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)





def build_dataloader(arch_list,batch_size, coder):
    x_data=[]
    y_data=[]
    for d in arch_list:
        x_data.append(coder.encode_gene(d[:-1]))
        y_data.append(d[-1])
    x_data = torch.tensor(x_data, dtype=torch.float)
    y_data = torch.tensor(y_data)
    # print('x_data: ',x_data.size())
    # print('y_data:',y_data.size())


    # random shuffle
    shuffle_idx = torch.randperm(len(x_data))
    x_data = x_data[shuffle_idx]
    y_data = y_data[shuffle_idx]

    # split data
    idx = x_data.size(0) // 5 * 4
    val_idx = x_data.size(0) // 5 * 4
    X_train, Y_train = x_data[:idx], y_data[:idx]
    X_test, Y_test = x_data[val_idx:], y_data[val_idx:]
    print('Train Size: %d,' % len(X_train), 'Valid Size: %d' % len(X_test))

    # build data loader
    train_dataset = RegDataset(X_train, Y_train)
    val_dataset = RegDataset(X_test, Y_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=2)
    return train_loader, valid_loader

