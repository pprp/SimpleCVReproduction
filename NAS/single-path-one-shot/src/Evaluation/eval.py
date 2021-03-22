import os
import torch
import pickle
from flops import get_cand_flops


def main():
    info = torch.load('../Search/log/checkpoint.pth.tar')['vis_dict']
    cands = sorted([cand for cand in info if 'err' in info[cand]],
                   key=lambda cand: info[cand]['err'])[:1]

    dst_dir = 'data'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for cand in cands:
        flops = get_cand_flops(cand)
        dst_path = os.path.join(dst_dir, str(cand))
        if os.path.exists(dst_path):
            continue
        print(cand, flops)
        print(cand, info[cand]['err'], flops)
        os.system('cp -r {} \'{}\''.format('template', dst_path))
        with open(os.path.join(dst_path, 'arch.pkl'), 'wb') as f:
            pickle.dump(cand, f)
if __name__ == '__main__':
    main()
