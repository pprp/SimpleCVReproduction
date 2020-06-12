import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

'''
数据集格式如下：
data
    - 1
        - a.jpg
        - b.jpg
        - c.jpg
    - 2
        - d.jpg
        - e.jpg
        - h.jpg
    - ...
1和2代表的是图片类别
'''
class tSNE_Visual():
    def __init__(self):
        super(tSNE_Visual, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            '--Input', type=str, default='data', help='the path of target dataset')
        self.parser.add_argument(
            '--Size', type=int, default=400, help='the size of every class')
        self.parser.add_argument('--Zoom', type=float,
                                 default=0.1, help='the size of every class')
        self.parser.add_argument(
            '--Output', type=str, default='t-SNE1.png', help='the out path of result image')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

    def plot_embedding(self, X, _output, zoom, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                # if np.min(dist) < 4e-3:
                # don't show points that are too close
                #   continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(
                        real_imgs[i], zoom=0.12, cmap=plt.cm.gray_r),
                    X[i], pad=0)
                ax.add_artist(imagebox)

        '''for i in range(X.shape[0]):
            #cls = plt.text(X[i, 0], X[i, 1], _classes[y[i][0].astype(int)-1],
            cls = plt.text(X[i, 0], X[i, 1], str(y[i].astype(int)),
            #cls = plt.text(X[i, 0], X[i, 1], '★',
                     color=_colors[int(y[i][0]-1)],
                     fontdict={'weight': 'bold', 'size': 12})
            cls.set_zorder(20) '''

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.savefig(_output)


if __name__ == '__main__':
    # Disable the GUI matplotlib
    plt.switch_backend('agg')

    tsne_visual = tSNE_Visual()
    opts = tsne_visual.parse()
    dataroot = opts.Input
    _size = opts.Size
    _output = opts.Output
    _zoom = opts.Zoom

    dirs = []
    for item in os.listdir(dataroot):
        if('.ipynb_checkpoints' not in item):
            dirs.append(item)

    _len = len(dirs)
    y = np.zeros((_size * _len, 1))
    for i in range(_len):
        y[i * _size: (i+1) * _size] = i+1

    imgs = []
    real_imgs = []
    for i in range(_len):
        single_cls = []
        path = os.path.join(dataroot, dirs[i])
        dataset_list = os.listdir(path)
        cnt = 0
        for item in dataset_list:
            if(cnt == _size):
                break
            if('.ipynb_checkpoints' in item):
                continue
            data_path = os.path.join(path, item)
            temp = cv2.imread(data_path)
            real_img = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
            imgs.append(temp.reshape(-1))
            real_imgs.append(real_img)
            cnt = cnt + 1
    np_imgs = np.array(imgs)
    real_imgs = np.array(real_imgs)

    tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    print(np_imgs.shape)
    result = tsne.fit_transform(np_imgs)

    tsne_visual.plot_embedding(X=result, _output=_output, zoom=_zoom)
