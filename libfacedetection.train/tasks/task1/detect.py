#!/usr/bin/python3
from __future__ import print_function
from yufacedetectnet import YuFaceDetectNet
from timer import Timer
from utils import decode
from nms import nms
from prior_box import PriorBox
from config import cfg

import os
import sys
import torch
import torch.backends.cudnn as cudnn
import argparse
import cv2
import numpy as np
from collections import OrderedDict

sys.path.append(os.getcwd() + '/../../src')


parser = argparse.ArgumentParser(description='Face and Landmark Detection')

parser.add_argument('-m', '--trained_model', default='weights/FaceBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--image_file', default='', type=str,
                    help='the image file to be detected')
parser.add_argument('--confidence_threshold', default=0.3,
                    type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3,
                    type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
#parser.add_argument('-s', '--show_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.3, type=float,
                    help='visualization_threshold')
parser.add_argument('--base_layers', default=16, type=int,
                    help='the number of the output of the first layer')
parser.add_argument('--device', default='cuda:0',
                    help='which device the program will run on. cuda:0, cuda:1, ...')
args = parser.parse_args()


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
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    #img_dim = 320
    device = torch.device(args.device)

    torch.set_grad_enabled(False)

    # net and model
    net = YuFaceDetectNet(phase='test', size=None)    # initialize detector
    net = load_model(net, args.trained_model, True)

    # net.load_state_dict(torch.load(args.trained_model))
    net.eval()

    print('Finished loading model!')

    # Print model's state_dict
    #print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    # print(net)

    cudnn.benchmark = True
    net = net.to(device)

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    # testing begin
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape

    #img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height])
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    print('there are', len(boxes), 'candidates')

    # for ss in scores:
    #    print('score', ss)
    # for bb in boxes:
    #    print('box', bb, bb[2]-bb[0], bb[3]-bb[1])

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 14])
    keep = nms(dets[:, selected_idx], args.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:args.keep_top_k, :]
    _t['misc'].toc()

    # save dets
    face_cc = 0
    for k in range(dets.shape[0]):
        if dets[k, 14] < args.vis_thres:
            continue
        xmin = dets[k, 0]
        ymin = dets[k, 1]
        xmax = dets[k, 2]
        ymax = dets[k, 3]
        score = dets[k, 14]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        print('{}: {:.3f} {:.3f} {:.3f} {:.3f} {:.10f}'.format(
            face_cc, xmin, ymin, w, h, score))
        face_cc = face_cc + 1

    # show image
    for b in dets:
        if b[14] < args.vis_thres:
            continue
        text = "{:.4f}".format(b[14])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)

        cv2.circle(img_raw, (b[4], b[4 + 1]), 2, (255, 0, 0), 2)
        cv2.circle(img_raw, (b[4 + 2], b[4 + 3]), 2, (0, 0, 255), 2)
        cv2.circle(img_raw, (b[4 + 4], b[4 + 5]), 2, (0, 255, 255), 2)
        cv2.circle(img_raw, (b[4 + 6], b[4 + 7]), 2, (255, 255, 0), 2)
        cv2.circle(img_raw, (b[4 + 8], b[4 + 9]), 2, (0, 255, 0), 2)

        cx = b[0]
        cy = b[1] + 12
        cv2.putText(img_raw, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('res', img_raw)
    cv2.resizeWindow('res', im_width, im_height)

    cv2.waitKey(0)
