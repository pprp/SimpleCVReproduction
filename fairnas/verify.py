from __future__ import print_function, division

import os
import argparse

import torch

from models.FairNAS_A import FairNasA
from models.FairNAS_B import FairNasB
from models.FairNAS_C import FairNasC
from dataloader import get_imagenet_dataset
from accuracy import accuracy

parser = argparse.ArgumentParser(description='FairNAS Config')
parser.add_argument('--model', default='FairNAS_A', choices=['FairNAS_A', 'FairNAS_B', 'FairNAS_C'])
parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'])
parser.add_argument('--val-dataset-root', default='/Your_Root/ILSVRC2012', help="val dataset root path")
parser.add_argument('--pretrained-path', default='./pretrained/FairNasA.pth.tar', help="checkpoint path")
parser.add_argument('--batch-size', default=256, type=int, help='val batch size')
parser.add_argument('--gpu-id', default=0, type=int, help='gpu to run')
args = parser.parse_args()

if __name__ == "__main__":
    assert args.model in ['FairNAS_A', 'FairNAS_B', 'FairNAS_C'], "Unknown model name %s" % args.model
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if args.model == "FairNAS_A":
        model = FairNasA()
    elif args.model == "FairNAS_B":
        model = FairNasB()
    elif args.model == "FairNAS_C":
        model = FairNasC()
    device = torch.device(args.device)
    pretrained_path = args.pretrained_path
    model_dict = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(model_dict["model_state"])
    if device.type == 'cuda':
        model.cuda()
    model.eval()

    val_dataloader = get_imagenet_dataset(batch_size=args.batch_size,
                                          dataset_root=args.val_dataset_root,
                                          dataset_tpye="valid")

    print("Start to evaluate ...")
    total_top1 = 0.0
    total_top5 = 0.0
    total_counter = 0.0
    for image, label in val_dataloader:
        image, label = image.to(device), label.to(device)
        result = model(image)
        top1, top5 = accuracy(result, label, topk=(1, 5))
        if device.type == 'cuda':
            total_counter += image.cpu().data.shape[0]
            total_top1 += top1.cpu().data.numpy()
            total_top5 += top5.cpu().data.numpy()
        else:
            total_counter += image.data.shape[0]
            total_top1 += top1.data.numpy()
            total_top5 += top5.data.numpy()
    mean_top1 = total_top1 / total_counter
    mean_top5 = total_top5 / total_counter
    print('Evaluate Result: Total: %d\tmTop1: %.4f\tmTop5: %.6f' % (total_counter, mean_top1, mean_top5))
