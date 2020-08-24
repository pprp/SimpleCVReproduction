
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float() # 找到极大值点
    return heat * keep 


def _topk(scores, K=40):
    # score shape : [batch, class , h, w]
    batch, cat, height, width = scores.size()

    # to shape: [batch , class, h * w] 分类别，每个class channel统计最大值
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    '''
    >>> torch.topk(torch.from_numpy(x), 3)
    torch.return_types.topk(
    values=tensor([[0.4863, 0.2339, 0.1153],
            [0.8370, 0.5961, 0.2733],
            [0.7303, 0.4656, 0.4475]], dtype=torch.float64),
    indices=tensor([[3, 2, 0],
            [1, 0, 2],
            [2, 1, 3]]))
    >>> x
    array([[0.11530714, 0.014376  , 0.23392263, 0.48629663],
        [0.59611302, 0.83697236, 0.27330404, 0.17728915],
        [0.36443852, 0.46562404, 0.73033529, 0.44751189]])
    '''

    topk_inds = topk_inds % (height * width)
    # 找到横纵坐标
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # to shape: [batch , class * h * w]
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)

    # 所有类别中找到最大值
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)

    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, K)

    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100):
    '''
    hmap提取中心点位置为xs,ys
    regs保存的是偏置，需要加在xs,ys上，代表精确的中心位置
    w_h_保存的是对应目标的宽和高
    '''
    # dets = ctdet_decode(*output, K=cfg.test_topk)
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap) # 归一化到0-1

    # if flip test
    if batch > 1: # batch > 1代表使用了翻转
        # img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]

    batch = 1

    # 这里的nms和带anchor的目标检测方法中的不一样，这里使用的是3x3的maxpool筛选
    hmap = _nms(hmap)  # perform nms on heatmaps
    
    # 找到前K个极大值点代表存在目标
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    # from [bs c h w] to [bs, h, w, c] 
    regs = _tranpose_and_gather_feature(regs, inds)

    regs = regs.view(batch, K, 2)

    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    # xs,ys是中心坐标，w_h_[...,0:1]是w,1:2是h
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections

