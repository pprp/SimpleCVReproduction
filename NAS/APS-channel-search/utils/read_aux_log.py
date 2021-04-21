import os
import re
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace as br

# download logs from seven
path = 'chan-search-enas/models/resnet_20_noproj_aux_chann_aux/YK0fFWqc'
path_suffix = '/'.join(path.split('/')[-2:])
local_path = os.path.join('./models/seven_models/', path_suffix )
if not os.path.isdir(local_path):
  os.makedirs(local_path)

if os.system('seven disk -download -src %s/record.log -dest %s' % (path, local_path)):
  print("Download logs from %s \t successfully" % path)

logdir = os.path.join(local_path, 'record.log')
num_blocks = 9
acc_0s = [] # [[accs over layers] accs over epochs]
acc_1s = []
acc_2s = []
acc_3s = []
std_0s = [] # [[accs over layers] accs over epochs]
std_1s = []
std_2s = []
std_3s = []

tmp_acc0 = [[] for i in range(num_blocks)] # [[accs over layers] accs over test archs]
tmp_acc1 = [[] for i in range(num_blocks)]
tmp_acc2 = [[] for i in range(num_blocks)]
tmp_acc3 = [[] for i in range(num_blocks)]

with open(logdir, 'r') as f:
  epoch = -1
  layer = -1
  for line in f.readlines():
    if 'Training at Epoch:' in line:
      epoch += 1
      if epoch != 0:
        # summary tmp_accs into accs
        mean_0 = [np.mean(v) for v in tmp_acc0]
        std_0 = [np.std(v) for v in tmp_acc0]
        mean_1 = [np.mean(v) for v in tmp_acc1]
        std_1 = [np.std(v) for v in tmp_acc1]
        mean_2 = [np.mean(v) for v in tmp_acc2]
        std_2 = [np.std(v) for v in tmp_acc2]
        mean_3 = [np.mean(v) for v in tmp_acc3]
        std_3 = [np.std(v) for v in tmp_acc3]
        acc_0s.append(mean_0)
        acc_1s.append(mean_1)
        acc_2s.append(mean_2)
        acc_3s.append(mean_3)
        std_0s.append(std_0)
        std_1s.append(std_1)
        std_2s.append(std_2)
        std_3s.append(std_3)

      # clear the tmp accs for the next epoch
      tmp_acc0 = [[] for i in range(num_blocks)] # [[accs over layers] accs over test archs]
      tmp_acc1 = [[] for i in range(num_blocks)]
      tmp_acc2 = [[] for i in range(num_blocks)]
      tmp_acc3 = [[] for i in range(num_blocks)]

    if 'layer: ' in line:
      layer += 1
      if layer == 9:
        # reset for the next cand arch network
        layer = 0

    if line.startswith('arch'):
      assert layer > -1 and layer < 9, "layer should be [0, 8]"

      pattern_0 = re.compile(r'arch: 16, acc: \d{1,3}\.\d{3} \|\|')
      pattern_1 = re.compile(r'arch: 32, acc: \d{1,3}\.\d{3} \|\|')
      pattern_2 = re.compile(r'arch: 64, acc: \d{1,3}\.\d{3} \|\|')
      pattern_3 = re.compile(r'arch: 96, acc: \d{1,3}\.\d{3} \|\|')

      substr_0 = pattern_0.search(line).group()
      tmp_acc0[layer].append(float(substr_0.split()[-2]))
      substr_1 = pattern_1.search(line).group()
      tmp_acc1[layer].append(float(substr_1.split()[-2]))
      substr_2 = pattern_2.search(line).group()
      tmp_acc2[layer].append(float(substr_2.split()[-2]))
      substr_3 = pattern_3.search(line).group()
      tmp_acc3[layer].append(float(substr_3.split()[-2]))

# start plot
# create plot fig
fig_dir = os.path.join('aux_accs/', path_suffix)
if not os.path.isdir(fig_dir):
  os.makedirs(fig_dir)

epoch_intvl = 5
epoch_idx = np.linspace(0, len(acc_0s)-1, num=len(acc_0s)//epoch_intvl, dtype=np.int64)
for layer in range(num_blocks):
  layer_acc0 = np.asarray(acc_0s)[epoch_idx, layer]
  layer_acc1 = np.asarray(acc_1s)[epoch_idx, layer]
  layer_acc2 = np.asarray(acc_2s)[epoch_idx, layer]
  layer_acc3 = np.asarray(acc_3s)[epoch_idx, layer]
  layer_std0 = np.asarray(std_0s)[epoch_idx, layer]
  layer_std1 = np.asarray(std_1s)[epoch_idx, layer]
  layer_std2 = np.asarray(std_2s)[epoch_idx, layer]
  layer_std3 = np.asarray(std_3s)[epoch_idx, layer]
  plt.figure(figsize=(30, 10))
  # plt.errorbar(epoch_idx, layer_acc0, label='arch 16')
  # plt.errorbar(epoch_idx, layer_acc1, label='arch 32')
  # plt.errorbar(epoch_idx, layer_acc2, label='arch 64')
  # plt.errorbar(epoch_idx, layer_acc3, label='arch 96')
  plt.errorbar(epoch_idx, layer_acc0, layer_std0, label='arch 16')
  plt.errorbar(epoch_idx, layer_acc1, layer_std1, label='arch 32')
  plt.errorbar(epoch_idx, layer_acc2, layer_std2, label='arch 64')
  plt.errorbar(epoch_idx, layer_acc3, layer_std3, label='arch 96')
  plt.legend(loc='lower right')
  plt.savefig(os.path.join(fig_dir, 'acc_layer_%d.png' % layer))
  plt.close()
pass

os.system('tar -cf ./aux_accs/aux_accs.tar.gz %s' % fig_dir)
os.system('sz ./aux_accs/aux_accs.tar.gz')
