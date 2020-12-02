# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:57:15 2020

@author: Lim
"""
import os
import sys
import torch
import numpy as np
from Loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset

sys.path.append(r'./backbone')
#from resnet import ResNet  
#from resnet_dcn import ResNet
#from dlanet import DlaNet
from dlanet_dcn import DlaNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0' 
use_gpu = torch.cuda.is_available()
#model = ResNet(34)
model = DlaNet(34)
print('cuda', torch.cuda.current_device(), torch.cuda.device_count())

loss_weight={'hm_weight':1,'wh_weight':0.1,'ang_weight':0.1,'reg_weight':0.1}
criterion = CtdetLoss(loss_weight)

device = torch.device("cuda")
if use_gpu:
    model.cuda()
model.train()

learning_rate = 1.25e-4
num_epochs = 150

# different learning rate
params=[]
params_dict = dict(model.named_parameters())
for key,value in params_dict.items():
    params += [{'params':[value],'lr':learning_rate}]

#optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)


train_dataset = ctDataset(split='train')
train_loader = DataLoader(train_dataset,batch_size=2,shuffle=False,num_workers=0)  # num_workers是加载数据（batch）的线程数目

test_dataset = ctDataset(split='val')
test_loader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=0)
print('the dataset has %d images' % (len(train_dataset)))


num_iter = 0

best_test_loss = np.inf 

for epoch in range(num_epochs):
    model.train()
    if epoch == 90:
        learning_rate= learning_rate * 0.1 
    if epoch == 120:
        learning_rate= learning_rate * (0.1 ** 2)
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
     
    total_loss = 0.
    
    for i, sample in enumerate(train_loader):
        for k in sample:
            sample[k] = sample[k].to(device=device, non_blocking=True)
        pred = model(sample['input'])
        loss = criterion(pred, sample)    
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1)))
            num_iter += 1
            

    #validation
    validation_loss = 0.0
    model.eval()
    for i, sample in enumerate(test_loader):
        if use_gpu:
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)
        
        pred = model(sample['input'])
        loss = criterion(pred, sample)   
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('get best test loss %.5f' % best_test_loss)
        torch.save(model.state_dict(),'best.pth')
    torch.save(model.state_dict(),'last.pth')
