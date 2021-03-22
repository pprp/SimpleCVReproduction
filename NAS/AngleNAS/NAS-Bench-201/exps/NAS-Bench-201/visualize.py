##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
# python3 exps/NAS-Bench-201/visualize.py --api_path $HOME/.torch/NAS-Bench-201-v1_0-e61699.pth
##################################################
import os, sys, time, argparse, collections
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('agg')
import matplotlib.pyplot as plt

lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from log_utils    import time_string
from nas_102_api  import NASBench102API as API
import pdb
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches

# ================== ABS ===================

def get_different_search_space_result(xlabel, ylabel, foldername, file_name):
  xxxstrs = ['DARTS','SETN', 'ENAS', 'GDAS', 'SPOS']
  datasets = ['cifar10','cifar100','ImageNet16-120']
  subsets = ['ori-test', 'x-test', 'x-test']
  x_maxs, y_limss =250, [(40, 100,10), (0, 80,10), (0, 50,10)]

  color_set = ['b', 'g', 'c', 'm', 'y', 'r']
  dpi, width, height = 300, 3000, 1400
  LabelSize, LegendFontsize = 16, 16
  figsize = width / float(dpi), height / float(dpi)

  font1 = {'family' : 'cmr10','weight' : 'medium', 'size' : 11}
  plt.rc('font', **font1)          # controls default text sizes
  plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
  plt.rc('axes', labelsize=LabelSize)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('legend', fontsize=LabelSize)    # legend fontsize
  plt.rc('figure', titlesize=LabelSize)


  all_accs = [[[70.92, 54.29666666666666, 88.67, 70.92, 70.92, 84.16333333333333, 84.61333333333333, 54.29666666666666, 88.9, 86.45333333333333, 89.37],
                [87.54333333333334, 87.71, 91.76666666666667, 88.89333333333332, 87.64999999999999, 93.07, 89.33, 88.32666666666667, 92.66, 86.56333333333333, 93.2],
                [54.29666666666666, 54.29666666666666, 49.019999999999996, 54.29666666666666, 54.29666666666666,54.29666666666666, 49.019999999999996, 54.29666666666666, 93.76, 54.29666666666666, 93.76],
                [93.36, 93.76, 93.15, 93.67333333333333, 89.5, 93.22, 93.64, 93.76, 93.60666666666667, 65.9, 93.47333333333334],
                [92.62, 90.02, 92.39, 91.32333333333332, 89.04333333333334, 92.94666666666667, 90.66, 92.74, 93.25, 60.326666666666675, 93.79]],

              [[58.12666665649414, 61.38999994812012, 62.31999992879232, 54.6399999633789, 57.75999996337891, 54.6399999633789, 65.98000002237956, 67.57333330281575, 62.18000004882813, 58.13999995727539, 65.70666659952799],
                [67.12000002441407, 65.12000002441407,68.12000002441407,68.12000002441407,66.12000002441407,69.12000002441407,66.12000002441407,70.12000002441407,70.12000002441407,50.12000002441407,70.12000002441407,],
                [15.606666656494141, 15.606666656494141, 10.199999990844727, 15.606666656494141, 15.606666656494141, 15.606666656494141, 10.199999990844727, 15.606666656494141, 71.10666661783854, 15.606666656494141, 71.10666661783854],
                [67.26000002441407, 65.88000001220703, 46.679999981689456, 67.26000002441407, 61.23999996948242, 62.69999997558594, 67.13999991455078, 70.35999993896485, 69.77999986572266, 34.660000006103516, 71.0099999633789],
                [65.15999991861979, 65.9733333211263, 68.47333332722981, 68.43999993896485, 66.06000000610352,69.1599999593099, 66.6, 71.95999993896484, 69.75999997151693, 29.800000006103517, 70.49999994303386]],

              [[18.411111069573295, 16.322222180684406, 33.199999954223635, 18.411111069573295, 18.411111069573295, 26.09999997965495, 27.433333321465387, 16.322222180684406, 32.03333325703939, 27.82222220357259, 35.47777772691515],
                [37.233333335876466, 38.49999993896484, 39.06666662597656, 26.366666639539933, 20.955555531819662, 45.2, 38.933333292643226, 46.33333329603406, 41.44444439358181, 28.399999969482423, 44.23333333333333],
                [16.322222180684406, 16.322222180684406, 9.349999990463257, 16.322222180684406, 16.322222180684406, 16.322222180684406, 9.349999990463257, 16.322222180684406, 41.44444440375434, 16.322222180684406, 35.47777772691515],
                [41.022222110324435, 41.44444440375434, 41.022222110324435, 41.022222110324435, 15.94444443342421, 28.26666664123535, 36.95555545043945, 45.13333322143555, 41.022222110324435, 13.399999986436633, 44.89999996439616],
                [41.377777737087676, 36.64444435289171, 42.63333337402344, 36.76666662597656, 35.233333297729494, 45.66666662597656, 39.18888878716363, 42.53333331298828, 44.24444444105361, 15.433333267211914, 44.68888877360026]]]

  all_accs_ = []
  for i, accs in enumerate(all_accs):
    all_accs_.append([])
    for j, acc in enumerate(accs):
      # Based on NAS-Bench-201, we design 11 shrunk search spaces of various size, whose no. ranges from 0 to 10.
      # Eight ones are chosen from them.
      acc = acc[0:3]+acc[4:7]+acc[9:]
      all_accs_[i].append(acc)
  all_accs = all_accs_
  print(all_accs)

  plt.style.use('fivethirtyeight')
  fig, axes = plt.subplots(3,1,figsize=(10, 8))
  # search_space = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
  search_space = ["S0", "S1: original search space of NAS-Bench-201", "S2: S2 to S8 are various subsets of S1", "S3", "S4", "S5", "S6", "S7", "S8"]
  new_all_accs = np.transpose(np.array(all_accs), (0, 2, 1))
  color_list = ['peru','orchid','deepskyblue']
  x = range(1,6)
  for i in range(8):
    if i < 2:
      axes[0].bar([j-0.35+0.1*i for j in x], new_all_accs[0][i], width=0.1, label=search_space[i+1], color=color_list[i], alpha=0.8)
    else:
      axes[0].bar([j-0.35+0.1*i for j in x], new_all_accs[0][i], width=0.1, label=search_space[i+1], alpha=0.8)
  axes[0].set_xticklabels(labels=[""]+xxxstrs)
  axes[0].set_ylim(47, 95)
  axes[0].set_ylabel(ylabel)
  axes[0].set_title('CIFAR-10', fontsize=16)

  for i in range(8):
    if i < 2:
      axes[1].bar([j-0.35+0.1*i for j in x], new_all_accs[1][i], width=0.1, color=color_list[i], alpha=0.8)
    else:
      axes[1].bar([j-0.35+0.1*i for j in x], new_all_accs[1][i], width=0.1, alpha=0.8)
  axes[1].set_ylabel(ylabel)
  axes[1].set_xticklabels(labels=[""]+xxxstrs)
  axes[1].set_title('CIFAR-100', fontsize=16)


  for i in range(8):
    if i < 2:
      axes[2].bar([j-0.35+0.1*i for j in x], new_all_accs[2][i], width=0.1, color=color_list[i], alpha=0.8)
    else:
      axes[2].bar([j-0.35+0.1*i for j in x], new_all_accs[2][i], width=0.1, alpha=0.8)
  axes[2].set_ylabel(ylabel)
  axes[2].set_xticklabels(labels=[""]+xxxstrs)
  axes[2].set_xlabel(xlabel)
  axes[2].set_title('ImageNet-16-120', fontsize=16)

  fig.subplots_adjust(bottom=0.2, hspace=0.6)
  legend = fig.legend(ncol=4, loc='lower center')
  legend.get_frame().set_facecolor('none')

  save_path = foldername / '{}'.format(file_name)
  print('save figure into {:}\n'.format(save_path))
  plt.savefig(str(save_path), dpi=300, bbox_inches='tight', format='png', transparent=True)

def plot_standalone_model_rank(xlabel, ylabel, foldername, file_name):
    dpi, width, height = 300, 4000, 1350
    LabelSize, LegendFontsize = 20, 20
    figsize = width / float(dpi), height / float(dpi)

    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : LabelSize,}

    plt.rc('font', **font1)                 # controls default text sizes
    plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
    plt.rc('axes', labelsize=LabelSize)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LabelSize)    # legend fontsize
    plt.rc('figure', titlesize=LabelSize)

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ticks = np.arange(0, 51, 10)
    axs[0].set_xticks(ticks)
    axs[0].set_xticklabels(ticks)
    axs[1].set_xticks(ticks)
    axs[1].set_xticklabels(ticks)
    axs[2].set_xticks(ticks)
    axs[2].set_xticklabels(ticks)
    axs[0].set_xticks(ticks)
    axs[0].set_yticklabels(ticks)
    axs[1].set_yticks(ticks)
    axs[1].set_yticklabels(ticks)
    axs[2].set_yticks(ticks)
    axs[2].set_yticklabels(ticks)
    for ax in axs.flat:
        ax.label_outer()

    ax1=axs[0]
    real_rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    angle_rank = [7, 5, 8, 6, 4, 1, 3, 10, 2, 9, 20, 18, 16, 19, 17, 11, 13, 15, 12, 14, 21, 25, 27, 28, 29, 23, 30, 24, 22, 26, 31, 34, 38, 33, 36, 40, 37, 32, 35, 39, 41, 42, 43, 48, 45, 44, 46, 47, 50, 49]
    ax1.scatter(real_rank, angle_rank, alpha=0.6)
    ax1.set_title('CIFAR-10')
    ax1.text(real_rank[-1]-23, 2, 'Tau=0.833', fontsize=20)
    ax1.set_xticks(np.arange(0, real_rank[-1]+1, 10))
    ax1.set_yticks(np.arange(0, real_rank[-1]+1, 10))

    ax2=axs[1]
    real_rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    angle_rank = [10, 3, 4, 9, 2, 1, 5, 7, 6, 8, 12, 20, 15, 11, 19, 18, 13, 14, 16, 17, 28, 22, 26, 24, 29, 30, 27, 25, 23, 21, 34, 39, 40, 36, 33, 35, 31, 37, 38, 32, 48, 44, 42, 43, 41, 45, 46, 47, 49, 50]
    ax2.scatter(real_rank, angle_rank, alpha=0.6)
    ax2.set_title('CIFAR-100')
    ax2.text(real_rank[-1]-23, 2, 'Tau=0.822', fontsize=20)
    ax2.set_xticks(np.arange(0, real_rank[-1]+1, 10))
    ax2.set_yticks(np.arange(0, real_rank[-1]+1, 10))

    ax3=axs[2]
    real_rank = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    angle_rank = [2, 9, 10, 6, 5, 8, 7, 1, 4, 3, 14, 18, 16, 15, 13, 19, 12, 20, 17, 11, 22, 23, 21, 28, 29, 27, 24, 25, 26, 30, 36, 32, 34, 31, 35, 38, 39, 33, 40, 37, 50, 43, 45, 46, 47, 48, 41, 49, 42, 44]
    ax3.scatter(real_rank, angle_rank, alpha=0.6)
    ax3.set_title('ImageNet-16-120') #0.851
    ax3.text(real_rank[-1]-23, 2, 'Tau=0.825', fontsize=20)
    ax3.set_xticks(np.arange(0, real_rank[-1]+1, 10))
    ax3.set_yticks(np.arange(0, real_rank[-1]+1, 10))
    fig.tight_layout()

    save_path = foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight', format='png')


def plot_ranking_stability(xlabel, ylabel, foldername,file_name):
  def get_data(dataset):
    cor_data = torch.load("exps/NAS-Bench-102/corr_data/rebn_{}_all_diff_seed_tau_dict.tau".format(dataset))

    acc_data = cor_data['acc']
    acc_no_rebn_data = cor_data['acc_no_rebn']
    angle_data = cor_data['angle']
    add_data = cor_data['add']
    random_data = cor_data['random']

    return random_data, acc_no_rebn_data, acc_data, angle_data

  def re_construct_data(data):
    data_X1 = []
    data_Y1 = []
    for key, value in data:
      data_X1.append(key)
      data_Y1.append(value)
    return data_Y1

  LabelSize, LegendFontsize = 25, 25
  font1 = {'family' : 'cmr10', 'weight' : 'medium', 'size'   : LabelSize}
  plt.rc('font', **font1)          # controls default text sizes
  plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
  plt.rc('axes', labelsize=LabelSize)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('legend', fontsize=LabelSize)    # legend fontsize
  plt.rc('figure', titlesize=LabelSize)
  plt.figure(figsize=(16, 4))

  random_data = []
  acc_no_rebn_data = []
  acc_data = []
  angle_data = []
  datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
  for dataset in datasets:
    tmp_random_data, tmp_acc_no_rebn_data, tmp_acc_data, tmp_angle_data = get_data(dataset)
    tmp_acc_no_rebn_data = sorted(tmp_acc_no_rebn_data.items(), key=lambda d:d[0])
    tmp_acc_data = sorted(tmp_acc_data.items(), key=lambda d:d[0])
    tmp_angle_data = sorted(tmp_angle_data.items(), key=lambda d:d[0])
    tmp_random_data = sorted(tmp_random_data.items(), key=lambda d:d[0])

    random_data.append(re_construct_data(tmp_random_data))
    acc_no_rebn_data.append(re_construct_data(tmp_acc_no_rebn_data))
    acc_data.append(re_construct_data(tmp_acc_data))
    angle_data.append(re_construct_data(tmp_angle_data))

  width = 0.20
  locations = list(range(len(acc_data)))
  locations = [i+1-0.135 for i in locations]

  positions1 = locations
  boxplot1 = plt.boxplot(acc_data, positions=positions1, patch_artist=True, showfliers=True, widths=width)

  positions2 = [x+(width+0.08) for x in locations]
  boxplot2 = plt.boxplot(angle_data, positions=positions2, patch_artist=True, showfliers=True, widths=width)

  for box in boxplot1['boxes']:
    box.set(color='#3c73a8')
  for box in boxplot2['boxes']:
    box.set(color='#fec615')

  plt.xlim(0, len(acc_data)+1)

  ticks = np.arange(0, len(acc_data)+1, 1)
  ticks_label_ = ['CIFAR-10', 'CIFAR-100', 'ImageNet-16-120', '']
  ticks_label, num = [], 0
  print(ticks_label_)
  print(ticks)
  for i in range(len(acc_data)+1):
    ticks_label.append(str(ticks_label_[i-1]))
  plt.xticks(ticks, ticks_label)#, rotation=45)
  plt.xlabel(xlabel, fontsize=25)
  plt.ylabel(ylabel, fontsize=25)

  plt.grid()
  plt.plot([], c='#3c73a8', label='Acc. w/ ReBN')
  plt.plot([], c='#fec615', label='Angle')
  legend = plt.legend(ncol=6, loc='lower center', bbox_to_anchor=(0.5, -0.55))

  save_path = foldername / '{}'.format(file_name)
  print('save figure into {:}\n'.format(save_path))
  plt.savefig(str(save_path), bbox_inches='tight', format='png')

# We perform multiple experiments with different seeds
def plot_corr_at_early_traing_stage(xlabel, ylabel, foldername,file_name):
    # CIFAR-10
    values_cifar10 = {}
    values_cifar10['angle'] = [[0.57,0.59,0.58,0.60,0.59],
                               [0.54,0.57,0.57,0.54,0.53],
                               [0.56,0.57,0.59,0.56,0.55],
                               [0.55,0.57,0.58,0.53,0.52],
                               [0.57,0.60,0.58,0.56,0.54]]

    values_cifar10['acc'] = [[-0.18,0.01,0.05,0.15,0.24],
                             [0.02,0.12,0.33,0.41,0.46],
                             [0.21,0.20,0.20,0.24,0.27],
                             [0.13,0.16,0.21,0.22,0.24],
                             [0.06,0.15,0.29,0.31,0.33]]

    # CIFAR-100
    values_cifar100 = {}
    values_cifar100['angle'] = [[0.46,0.41,0.53,0.61,0.63],
                                [0.42,0.38,0.52,0.60,0.62],
                                [0.46,0.48,0.51,0.61,0.63],
                                [0.42,0.47,0.53,0.61,0.63],
                                [0.41,0.43,0.54,0.60,0.62]]

    values_cifar100['acc'] = [[0.34, 0.44, 0.37, 0.50, 0.49],
                              [0.23, 0.42, 0.39, 0.48, 0.53],
                              [0.20, 0.30, 0.45, 0.50, 0.45],
                              [0.11, 0.40, 0.45, 0.47, 0.47],
                              [0.17, 0.35, 0.46, 0.43, 0.47]]


    # Imagenet
    values_imagenet = {}
    values_imagenet['angle'] = [[0.24,0.49,0.51,0.52,0.53],
                                [0.24,0.49,0.51,0.52,0.54],
                                [0.24,0.49,0.50,0.52,0.53],
                                [0.25,0.49,0.50,0.51,0.52],
                                [0.25,0.50,0.51,0.53,0.53]]


    values_imagenet['acc'] = [[0.09, 0.33, 0.46, 0.44,0.44],
                              [0.11, 0.20, 0.40, 0.47,0.54],
                              [0.06, 0.26, 0.37, 0.43,0.40],
                              [0.13, 0.32, 0.45, 0.38,0.49],
                              [0.17, 0.29, 0.36, 0.44,0.34]]

    epochs = [5,15,25,35,45]
    color_set = ['y','b']
    dpi, width, height = 300, 3000, 1400
    LabelSize, LegendFontsize = 20, 20
    figsize = width / float(dpi), height / float(dpi)

    font1 = {'family' : 'Times New Roman','weight' : 'medium', 'size' : LabelSize}
    plt.rc('font', **font1)          # controls default text sizes
    plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
    plt.rc('axes', labelsize=LabelSize)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LabelSize)    # legend fontsize
    plt.rc('figure', titlesize=LabelSize)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axs[0]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('CIFAR-10')
    ax.set_ylabel(ylabel)
    epochs_cifar = [2,4,6,8,10]
    mean_angle, std_angle = np.mean(np.array(values_cifar10['angle']), axis=0), np.std(np.array(values_cifar10['angle']), axis=0) / np.sqrt(len(values_cifar10['angle']))
    mean_acc, std_acc = np.mean(np.array(values_cifar10['acc']), axis=0), np.std(np.array(values_cifar10['acc']), axis=0) / np.sqrt(len(values_cifar10['acc']))
    ax.plot(epochs_cifar, mean_angle, color=color_set[0], linestyle='-', label='Angle', lw=3.0)
    ax.fill_between(epochs_cifar, mean_angle - std_angle, mean_angle + std_angle, alpha=0.3)
    ax.plot(epochs_cifar, mean_acc, color=color_set[1], linestyle='-', label='Acc. w / ReBN', lw=2.0)
    ax.fill_between(epochs_cifar, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)

    ax = axs[1]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('CIFAR-100')
    mean_angle, std_angle = np.mean(np.array(values_cifar100['angle']), axis=0), np.std(np.array(values_cifar100['angle']), axis=0) / np.sqrt(len(values_cifar100['angle']))
    mean_acc, std_acc = np.mean(np.array(values_cifar100['acc']), axis=0), np.std(np.array(values_cifar100['acc']), axis=0) / np.sqrt(len(values_cifar100['acc']))
    ax.plot(epochs, mean_angle, color=color_set[0], linestyle='-', lw=3.0)
    ax.fill_between(epochs, mean_angle - std_angle, mean_angle + std_angle, alpha=0.3)
    ax.plot(epochs, mean_acc, color=color_set[1], linestyle='-', lw=2.0)
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)
    font1 = {'family' : 'Times New Roman','weight' : 'medium', 'size' : 18}
    plt.rc('font', **font1)
    ax.annotate('standard deviation', xy=(0.5, 0.52), xycoords='axes fraction', xytext=(0.3, 0.25), 
            arrowprops=dict(arrowstyle="->", color='r'))

    ax = axs[2]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('ImageNet-16-120')
    mean_angle, std_angle = np.mean(np.array(values_imagenet['angle']), axis=0), np.std(np.array(values_imagenet['angle']), axis=0) / np.sqrt(len(values_imagenet['angle']))
    mean_acc, std_acc = np.mean(np.array(values_imagenet['acc']), axis=0), np.std(np.array(values_imagenet['acc']), axis=0) / np.sqrt(len(values_imagenet['acc']))
    ax.plot(epochs, mean_angle, color=color_set[0], linestyle='-', lw=3.0)
    ax.fill_between(epochs, mean_angle - std_angle, mean_angle + std_angle, alpha=0.3)
    ax.plot(epochs, mean_acc, color=color_set[1], linestyle='-', lw=2.0)
    ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    fig.legend(ncol=2, loc='lower center')

    save_path = foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight', format='png')

# Angle-based search space shrinking
# Comparison with different metrics
def show_ABS(foldername, xlabel, ylabel, file_name):

  LabelSize = 12
  op_number = list(range(1,31))
  font1 = {'family' : 'Times New Roman','weight' : 'medium', 'size' : LabelSize}
  plt.rc('font', **font1)                 # controls default text sizes
  plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
  plt.rc('axes', labelsize=LabelSize)     # fontsize of the x and y labels
  plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
  plt.rc('legend', fontsize=LabelSize)    # legend fontsize
  plt.rc('figure', titlesize=LabelSize)

  # Repeated Experiment 1
  ranks = [[25, 22, 24, 21, 23, 20, 28, 29, 26, 18, 27, 11, 30, 14, 15, 17, 16, 19, 13, 9],
          [28, 29, 30, 26, 27, 3, 18, 7, 1, 10, 9, 12, 6, 13, 19, 17, 16, 8, 15, 2],
          [25, 12, 10, 30, 23, 13, 8, 24, 22, 3, 20, 9, 27, 7, 28, 21, 4, 26, 5, 29]]

  mask = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

  k, drop_num = 0, 20
  for rank, m, in zip(ranks, mask):
    if k > drop_num:
      break
    for i, r in enumerate(rank):
      m[r-1] = 1
    k+=1
  methods = ['angle', 'accuracy', 'magnitude']
  fig, axes = plt.subplots(3,1, figsize=(10, 4.2))
  df = pd.DataFrame(np.array(mask), index=methods)
  sns.heatmap(df,annot=False,cmap='summer',linewidths=3, cbar=False, center=0.5, ax=axes[0], xticklabels=False) #cbar_kws={"orientation":"horizontal"},
  axes[0].set_ylabel(ylabel)
  axes[0].set_title('Repeated Experiment 1')

  # Repeated Experiment 2
  ranks = [[25, 22, 24, 21, 23, 20, 28, 29, 26, 27, 18, 11, 30, 14, 15, 17, 16, 19, 13, 9],
          [28, 29, 30, 26, 27, 3, 18, 1, 19, 20, 10, 13, 9, 16, 23, 17, 21, 5, 7, 15],
          [25, 12, 10, 24, 13, 23, 30, 8, 22, 7, 3, 20, 21, 9, 5, 4, 27, 28, 26, 6]]

  mask = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

  k = 0
  for rank, m, in zip(ranks, mask):
    if k > drop_num:
      break
    for i, r in enumerate(rank):
      m[r-1] = 1
    k+=1
  df = pd.DataFrame(np.array(mask), index=methods)
  sns.heatmap(df,annot=False,cmap='summer',linewidths=3, cbar=False, center=0.5, ax=axes[1], xticklabels=False) #cbar_kws={"orientation":"horizontal"},
  axes[1].set_ylabel(ylabel)
  axes[1].set_title('Repeated Experiment 2')

  # Repeated Experiment 3
  ranks = [[25, 22, 24, 21, 23, 20, 28, 29, 26, 27, 18, 11, 30, 14, 15, 17, 16, 19, 13, 9],
          [28, 29, 30, 26, 27, 18, 3, 1, 19, 20, 13, 16, 9, 10, 17, 8, 23, 7, 14, 5],
          [25, 10, 13, 12, 23, 22, 24, 7, 8, 30, 3, 20, 5, 21, 4, 27, 9, 28, 26, 2]]

  mask = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

  k = 0
  for rank, m, in zip(ranks, mask):
    if k > drop_num:
      break
    for i, r in enumerate(rank):
      m[r-1] = 1
    k+=1
  df = pd.DataFrame(np.array(mask), index=methods, columns=op_number)
  sns.heatmap(df,annot=False,cmap='summer',linewidths=3, cbar=False, center=0.5, ax=axes[2], xticklabels=True) #cbar_kws={"orientation":"horizontal"},
  axes[2].set_xlabel(xlabel)
  axes[2].set_ylabel(ylabel)
  axes[2].set_title('Repeated Experiment 3')

  patches_labels = ['Reserved Operators', 'Removed Operators']
  color = ['#048243', '#fffe71']
  patches = [ mpatches.Patch(color=color[i], label="{:s}".format(patches_labels[i]) ) for i in range(len(color)) ] 
  fig.subplots_adjust(bottom=0.22, hspace = 0.3) 
  fig.legend(handles=patches, ncol=2, loc='lower center')

  save_path = foldername / '{}'.format(file_name)
  print('save figure into {:}\n'.format(save_path))
  plt.savefig(str(save_path), dpi=300, bbox_inches='tight', format='png')

def plot_angle_evolution(xlabel, ylabel, foldername,file_name):
    angles_cifar10 = [0.0, 0.14, 0.22, 0.29, 0.35, 0.42, 0.48, 0.53, 0.58, 0.63, 0.67, 0.71, 0.75, 0.79, 0.82, 0.85, 0.88, 0.91, 0.93, 0.96, 0.98, 1.0, 1.02, 1.04, 1.05, 1.07, 1.08, 1.1, 1.11, 1.12, 1.13, 1.15, 1.16, 1.17, 1.17, 1.18, 1.19, 1.2, 1.21, 1.21, 1.22, 1.23, 1.23, 1.24, 1.24, 1.25, 1.25, 1.26, 1.26, 1.26, 1.27, 1.27, 1.27, 1.28, 1.28, 1.28, 1.28, 1.29, 1.29, 1.29, 1.29, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31, 1.31]
    angles_cifar100 = [0.0, 0.22, 0.37, 0.51, 0.64, 0.75, 0.84, 0.91, 0.97, 1.02, 1.07, 1.1, 1.14, 1.16, 1.19, 1.21, 1.23, 1.25, 1.26, 1.27, 1.28, 1.3, 1.3, 1.31, 1.32, 1.33, 1.33, 1.34, 1.34, 1.35, 1.35, 1.36, 1.36, 1.36, 1.37, 1.37, 1.37, 1.38, 1.38, 1.38, 1.38, 1.38, 1.38, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.39, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4]
    angles_imagenet = [0.0, 0.37, 0.64, 0.86, 1.01, 1.11, 1.19, 1.24, 1.28, 1.31, 1.33, 1.35, 1.36, 1.37, 1.38, 1.39, 1.39, 1.4, 1.4, 1.41, 1.41, 1.42, 1.42, 1.42, 1.42, 1.43, 1.43, 1.43, 1.43, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.44, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45, 1.46, 1.46, 1.46, 1.45, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46, 1.46]

    color_set = ['y', 'b']
    dpi, width, height = 300, 3000, 1400
    LabelSize, LegendFontsize = 20, 20
    figsize = width / float(dpi), height / float(dpi)

    font1 = {'family' : 'Times New Roman','weight' : 'medium', 'size' : LabelSize}
    plt.rc('font', **font1)                 # controls default text sizes
    plt.rc('axes', titlesize=LabelSize)     # fontsize of the axes title
    plt.rc('axes', labelsize=LabelSize)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=LabelSize)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LabelSize)    # legend fontsize
    plt.rc('figure', titlesize=LabelSize)
    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))

    ax = axs[0]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('CIFAR-10')
    ax.set_ylabel(ylabel)
    epochs = list(range(100))
    ax.plot(epochs, angles_cifar10, color=color_set[0], linestyle='-', lw=2.5)
    ax.legend(loc='lower right', fontsize=15)

    ax = axs[1]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('CIFAR-100')
    ax.set_ylabel(ylabel)
    epochs = list(range(100))
    ax.plot(epochs, angles_cifar100, color=color_set[0], linestyle='-', lw=2.5)
    ax.legend(loc='lower right', fontsize=15)

    ax = axs[2]
    ax.grid()
    ax.set_xlabel(xlabel)
    ax.set_title('ImageNet-16-120')
    ax.set_ylabel(ylabel)
    epochs = list(range(100))
    ax.plot(epochs, angles_imagenet, color=color_set[0], linestyle='-', lw=2.5)
    ax.legend(loc='lower right', fontsize=15)
    fig.tight_layout()

    save_path = foldername / '{}'.format(file_name)
    print('save figure into {:}\n'.format(save_path))
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight', format='png')

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='NAS-Bench-102', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--save_dir',  type=str, default='./output/search-cell-nas-bench-102/visuals', help='The base-name of folder to save checkpoints and log.')
  parser.add_argument('--api_path',  type=str, default=None,                                         help='The path to the NAS-Bench-102 benchmark file.')
  args = parser.parse_args()
  
  vis_save_dir = Path(args.save_dir)
  vis_save_dir.mkdir(parents=True, exist_ok=True)
  meta_file = Path(args.api_path)
  assert meta_file.exists(), 'invalid path for api : {:}'.format(meta_file)

  api = API(args.api_path)
  
  # Fig. 4
  plot_standalone_model_rank(
        xlabel='Ranking at ground-truth setting', ylabel='Stand-alone \nRanking by angle',
        foldername=vis_save_dir, file_name='standalone_ranks.png')

  # Fig. 5
  plot_ranking_stability(xlabel='Datasets', ylabel='Ranking Correlation', 
        foldername=vis_save_dir, file_name='ranking_stability.png')

  # Fig. 6
  plot_corr_at_early_traing_stage(xlabel='Searching Epoch', ylabel='Ranking Correlation', \
        foldername=vis_save_dir, file_name='correlation_early_traing_stage.png')

  # Fig. 7
  show_ABS(
        foldername=vis_save_dir, xlabel='Ranking of Operators at Ground-Truth Setting (CIFAR-10)', \
        ylabel='Metrics', file_name='shrinkage_comparision.png')

  # Fig. 4 (Appendix)
  plot_angle_evolution( 
        xlabel='Epoch', ylabel='Angle',
        foldername=vis_save_dir, file_name='angle_evolution_between_epochs.png')

  # Fig. 2
  get_different_search_space_result(
        xlabel='NAS algorithms',
        ylabel='Top-1 Accuracy',
        foldername=vis_save_dir, file_name='different_search_space_visual.png')
  