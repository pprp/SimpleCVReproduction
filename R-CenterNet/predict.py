# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:20:07 2020

@author: Lim
"""
import os
import cv2
import math
import time
import torch
import numpy as np
import torch.nn as nn
from resnet_dcn import ResNet
from dlanet_dcn import DlaNet
from Loss import _gather_feat
from PIL import Image, ImageDraw
from dataset import get_affine_transform
from Loss import _transpose_and_gather_feat


def draw(filename,result):
    img = Image.open(filename)
    w, h=img.size
    draw = ImageDraw.Draw(img)
    for class_name,lx,ly,rx,ry,ang, prob in res:
        result = [int((rx+lx)/2),int((ry+ly)/2),int(rx-lx),int(ry-ly),ang]
        result=np.array(result)
        x=int(result[0])
        y=int(result[1])
        height=int(result[2])
        width=int(result[3])
        anglePi = result[4]/180 * math.pi
        anglePi = anglePi if anglePi <= math.pi else anglePi - math.pi
 
        cosA = math.cos(anglePi)
        sinA = math.sin(anglePi)
        
        x1=x-0.5*width   
        y1=y-0.5*height
        
        x0=x+0.5*width 
        y0=y1
        
        x2=x1            
        y2=y+0.5*height 
        
        x3=x0   
        y3=y2
        
        x0n= (x0 -x)*cosA -(y0 - y)*sinA + x
        y0n = (x0-x)*sinA + (y0 - y)*cosA + y
        
        x1n= (x1 -x)*cosA -(y1 - y)*sinA + x
        y1n = (x1-x)*sinA + (y1 - y)*cosA + y
        
        x2n= (x2 -x)*cosA -(y2 - y)*sinA + x
        y2n = (x2-x)*sinA + (y2 - y)*cosA + y
        
        x3n= (x3 -x)*cosA -(y3 - y)*sinA + x
        y3n = (x3-x)*sinA + (y3 - y)*cosA + y

        draw.line([(x0n, y0n),(x1n, y1n)], fill=(0, 0, 255),width=5) # blue  横线
        draw.line([(x1n, y1n),(x2n, y2n)], fill=(255, 0, 0),width=5) # red    竖线
        draw.line([(x2n, y2n),(x3n, y3n)],fill= (0,0,255),width=5)
        draw.line([(x0n, y0n), (x3n, y3n)],fill=(255,0,0),width=5)
#    plt.imshow(img)
#    plt.show()
    img.save(os.path.join('img_ret','dla_dcn_34_best_v2',os.path.split(filename)[-1]))

def pre_process(image):
    height, width = image.shape[0:2]
    inp_height, inp_width = 512, 512
    c = np.array([height / 2., width / 2.], dtype=np.float32)
    s = max(height, width) * 1.0
    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    inp_image = cv2.warpAffine(image, trans_input, (inp_width, inp_height),flags=cv2.INTER_LINEAR)

    mean = np.array([0.5194416012442385,0.5378052387430711,0.533462090585746], dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.3001546018824507, 0.28620901391179554, 0.3014112676161966], dtype=np.float32).reshape(1, 1, 3)
    
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width) # 三维reshape到4维，（1，3，512，512） 
    
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // 4, 
            'out_width': inp_width // 4}
    return images, meta


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float() 
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, ang, reg=None, K=100):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
   
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    
    ang = _transpose_and_gather_feat(ang, inds)
    ang = ang.view(batch, K, 1)

    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2,
                        ang], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def process(images, return_time=False):
    with torch.no_grad():
      output = model(images)
      hm = output['hm'].sigmoid_()
      ang = output['ang'].relu_()
      wh = output['wh']
      reg = output['reg'] 
      torch.cuda.synchronize()
      forward_time = time.time()
      dets = ctdet_decode(hm, wh, ang, reg=reg, K=100) # K 是最多保留几个目标
    if return_time:
      return output, dets, forward_time
    else:
      return output, dets


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:6].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def post_process(dets, meta):  
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])  
    num_classes = 1
    dets = ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
    for j in range(1, num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 6)
      dets[0][j][:, :5] /= 1
    return dets[0]


def merge_outputs(detections):
    num_classes = 1
    max_obj_per_img = 100
    scores = np.hstack([detections[j][:, 5] for j in range(1, num_classes + 1)])
    if len(scores) > max_obj_per_img:
      kth = len(scores) - max_obj_per_img
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, 2 + 1):
        keep_inds = (detections[j][:, 5] >= thresh)
        detections[j] = detections[j][keep_inds]
    return detections



if __name__ == '__main__':
#    model = ResNet(18)
    model = DlaNet(34)
    device = torch.device('cuda')
    
    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    model.cuda()
    for image_name in [os.path.join('imgs',f) for f in os.listdir('imgs')]:
#        image_name = 'data/images/011.jpg'
        if image_name.split('.')[-1] == 'jpg':
            image = cv2.imread(image_name)
            images, meta = pre_process(image)
            images = images.to(device)
            output, dets, forward_time = process(images, return_time=True)
            
            dets = post_process(dets, meta)
            ret = merge_outputs(dets)
            
            res = np.empty([1,7])
            for i, c in ret.items():
                tmp_s = ret[i][ret[i][:,5]>0.3]
                tmp_c = np.ones(len(tmp_s)) * (i+1)
                tmp = np.c_[tmp_c,tmp_s]
                res = np.append(res,tmp,axis=0)
            res = np.delete(res, 0, 0)
            res = res.tolist()
            draw(image_name, res)  # 画旋转矩形