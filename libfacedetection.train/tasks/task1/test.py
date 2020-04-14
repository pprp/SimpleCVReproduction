#!/usr/bin/python3
from __future__ import print_function

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd() + '/../../src')

from config import cfg
from prior_box import PriorBox
from nms import nms
from utils import decode
from timer import Timer
from yufacedetectnet import YuFaceDetectNet

from tqdm import tqdm
from datasets import WIDERFace


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu=True):
    # NOTE: load_to_cpu is to avoid GPU RAM surge when loading a model checkpoint,
    #       see https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load.
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def bbox_vote(det, nms_thresh=0.3):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    if det.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
        det = np.empty(shape=[0, 5])
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= nms_thresh)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        det_accu_sum[:, 4] = max_score
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum

    dets = dets[0:750, :]
    return dets

def simple_nms(dets, top_k=5000, nms_thresh=0.3):
    if dets.shape[0] == 0:
        dets = np.array([[10, 10, 20, 20, 0.002]])
    # keep top-K before NMS
    order = dets[:, -1].argsort()[::-1][:top_k]
    dets = dets[order, :]

    # do NMS
    keep = nms(dets, nms_thresh)
    dets = dets[keep, :]
    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]

    return dets

def detect_face(net, img, device, scale=1., conf_thresh=0.3):
    # set input x
    if scale != 1:
        img = cv2.resize(img, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    _, _, height, width = x.shape
    if device.type == 'cuda':
        x = x.to(device)

    # forward pass
    loc, conf = net(x)

    # get bounding boxes from PriorBox layer
    bbox_scale = torch.Tensor([width, height, width, height])
    priorbox = PriorBox(cfg, image_size=(height, width))
    priors = priorbox.forward()
    boxes = decode(loc.squeeze(0).data.cpu(), priors.data, cfg['variance'])
    boxes = boxes[:, :4] # omit landmarks
    boxes = boxes * bbox_scale / scale
    boxes = boxes.cpu().numpy()
    # get scores
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    # ignore low scores
    keep_ind = np.where(dets[:, -1] > conf_thresh)[0]
    dets = dets[keep_ind, :]
    return dets

def save_res(dets, event, name):
    txt_name = name[:-4]+'.txt'
    save_path = os.path.join(args.res_dir, event)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, txt_name), 'w') as f:
        f.write('{}\n'.format('/'.join([event, name])))
        f.write('{}\n'.format(dets.shape[0]))
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(np.floor(xmin), 
                                                                  np.floor(ymin), 
                                                                  np.ceil(w), 
                                                                  np.ceil(h), 
                                                                  score))

def get_available_scales(h, w, scales):
    smin = min(h, w)
    available_scales = []
    for scale in scales:
        if int(smin * scale) >= 64:
            available_scales.append(scale)
    return available_scales

def main(args):
    torch.set_grad_enabled(False)

    device = torch.device(args.device)

    # Initialize the net and load the model
    print('Loading pretrained model from {}'.format(args.trained_model))
    net = YuFaceDetectNet(phase='test', size=None)
    net = load_model(net, args.trained_model)
    net.eval()
    if device.type == 'cuda':
        cudnn.benchmark = True
        net = net.to(device)
    print('Finished loading model!')

    # init data loader for WIDER Face
    print('Loading data for {}...'.format(args.widerface_split))
    widerface = WIDERFace(args.widerface_root, split=args.widerface_split)
    print('Finished loading data!')

    # start testing
    scales = [1.]
    if args.multi_scale:
        scales = [0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.0]
    print('Performing testing with scales: {}, conf_threshold: {}'.format(str(scales), args.confidence_threshold))
    for idx in tqdm(range(len(widerface))):
        img, event, name = widerface[idx] # img_subpath = '0--Parade/XXX.jpg'

        dets = detect_face(net, img, device, conf_thresh=args.confidence_threshold)
        available_scales = get_available_scales(img.shape[0], img.shape[1], scales)
        for available_scale in available_scales:
            det = detect_face(net, img, device, scale=available_scale, conf_thresh=args.confidence_threshold)
            if det is not None: dets = np.row_stack((dets, det))

        # nms
        dets = simple_nms(dets)
        # dets = bbox_vote(dets)

        save_res(dets, event, name)

if __name__ == '__main__':
    def str2bool(v): # https://stackoverflow.com/a/43357954/6769366
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--widerface_root', default='./widerface', type=str, help='Path to WIDER Face root')
    parser.add_argument('--widerface_split', default='val', type=str, help='Either val or test.', choices=['val', 'test'])
    parser.add_argument('--res_dir', default='./results', type=str, help='Path to save evaluation results.')
    parser.add_argument('--multi_scale', default=False, type=str2bool, help='Use multi-scale testing or not. Default: False.')
    parser.add_argument('-m', '--trained_model', default='./weights/yunet_final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--confidence_threshold', default=0.3, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:1', help='which device the program will run on. cuda:0, cuda:1, ...')
    args = parser.parse_args()

    main(args)