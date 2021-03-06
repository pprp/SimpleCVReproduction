import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import CIFAR10

from accuracy import accuracy
from thop import profile
from models.timm_models import Fairnas


parser = argparse.ArgumentParser("Transfer Model Evaluation")
parser.add_argument('--model', type=str, required=True, help='transfer model')
parser.add_argument('--model-path', type=str, required=True, help='path to model')
parser.add_argument('--dataset', type=str, default='cifar10', help='transfer dataset')
parser.add_argument('--dataset-path', type=str, default='/home/work/dataset/cifar', help='dataset dir')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--se-ratio', default=1.0, type=float, help='squeeze-and-excitation ratio')
parser.add_argument('--epochs', type=int, default=200, help='fine-tune epochs')
parser.add_argument('--gpu_id', type=int, default=0, required=True, help='gpu id')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--dropout', type=float, default=0.1, metavar='DROP', help='Dropout rate (default: 0.)')


args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

# imagenet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(MEAN, STD),
])


val_data = CIFAR10(root=args.dataset_path, train=False, download=True, transform=val_transform)
val_quene = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                        pin_memory=True, num_workers=8)

# model
assert args.model in ['fairnas_a', 'fairnas_b', 'fairnas_c']
print(args.model)

model = Fairnas[args.model](s_r=args.se_ratio)

model.classifier = nn.Sequential(
					    nn.Dropout(args.dropout),  
						nn.Linear(1280, 10),
						)
model.load_state_dict(torch.load(args.model_path)['model_state'])


model = model.to(device)
input = torch.randn(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(input,), verbose=False)
print('flops: {}M, params: {}M'.format(flops / 1e6, params / 1e6))

model.eval()
criterion = nn.CrossEntropyLoss().to(device)
total_top1 = 0.0
total_top5 = 0.0
total_counter = 0.0
loss_ = 0.
with torch.no_grad():
	for step, (inputs, labels) in enumerate(val_quene):
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		loss = criterion(outputs, labels).mean()
		loss_ += loss
		top1, top5 = accuracy(outputs, labels, topk=(1, 5))
		if device.type == 'cuda':
			total_counter += inputs.cpu().data.shape[0]
			total_top1 += top1.cpu().data.numpy()
			total_top5 += top5.cpu().data.numpy()
		else:
			total_counter += inputs.data.shape[0]
			total_top1 += top1.data.numpy()
			total_top5 += top5.data.numpy()
	mean_top1 = total_top1 / total_counter
	mean_top5 = total_top5 / total_counter

print('Val. loss: {}, top1: {}, top5: {}'.format(loss_ / (step + 1), mean_top1, mean_top5))
