from __future__ import print_function, division
import argparse
import torch
from thop import clever_format, profile

from models.timm_models import Fairnas
from dataloader import get_imagenet_dataset
from accuracy import accuracy

def get_args():
    parser = argparse.ArgumentParser("Evaluate FairNAS-SE Models")
    parser.add_argument('--model', type=str, required=True, help='model to evaluate')
    parser.add_argument('--model-path', type=str, default='./pretrained/fairnas_a_se.pth.tar', help='dir to models')
    parser.add_argument('--se-ratio', default=1.0, type=float, help='squeeze-and-excitation ratio')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', default=256, type=int, help='val batch size')
    parser.add_argument('--val-dataset-root', default='/Your_Root/ILSVRC2012', help="val dataset root path")
    args = parser.parse_args()
    return args

def evaluate(args):
    model = Fairnas[args.model](s_r=args.se_ratio)
    device = torch.device(args.device)
    if args.device == 'cuda':
        model.cuda()
    state = torch.load(f'{args.model_path}',  map_location=device)
    model.load_state_dict(state)

    _input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(_input,), verbose=False)
    print('Model: {}, params: {}M, flops: {}M'.format(args.model, params / 1e6, flops / 1e6))
    
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



if __name__ == '__main__':
    args = get_args()
    evaluate(args)
