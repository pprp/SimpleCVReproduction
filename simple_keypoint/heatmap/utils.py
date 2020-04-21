import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def compute_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (C x H/r x W/r)
        targets (C x H/r x W/r)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    # beta=4
    neg_weights = torch.pow(1 - targets, 4).float()

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # 正样本

        # alpha=2
        # print(type(neg_weights))

        neg_loss = torch.log(1 - pred) * torch.pow(pred,2) * neg_weights * neg_inds # 负样本

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    width, height = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap


if __name__ == "__main__":
    # h w
    heatmap = np.zeros((360, 480))
    draw_umich_gaussian(heatmap, (240, 180), 20)

    print(heatmap.shape)

    plt.figure()
    img = plt.imshow(heatmap)
    img.set_cmap('hot')
    plt.savefig("heatmap.jpg")
