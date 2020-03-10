# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        ##标注计算类别范围以内的数，不在范围内的不考虑
        mask = (label_true >= 0) & (label_true < n_class)
        # 经过这句话的处理，行数将代表标签的值，而列数将代表预测的值，而具体的值代表个数。
        # 也就是说，对角线上的数分别指代第n个类别中，预测值和标签值相同的个数
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                     lp.flatten(),
                                                     self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix

        acc = np.diag(hist).sum() / hist.sum()  #总体精确度计算
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-6)  #求得每一类像素精确度的平均值
        acc_cls = np.nanmean(acc_cls)  #求得总体平均精确度
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) -
                              np.diag(hist) + 1e-6)  #按类别求得交并比
        mean_iu = np.nanmean(iu)  #求得总体平均交并比
        freq = hist.sum(axis=1) / hist.sum()  #求得标签中各个类别所占的比例
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()  #按权值计算交并比
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
