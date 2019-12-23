import os
import numpy as np

import torch
import torch.utils.data as data
import cv2

mean = [ 0.9010, 0.9049, 0.9025]
std = [ 0.1521, 0.1347, 0.1458]

class CaptchaDataset(data.Dataset):
    """ Captcha dataset warpper
    """
    def __init__(self, root_dir, mean, std):
        super(CaptchaDataset, self).__init__()
        if not os.path.exists(root_dir):
            raise RuntimeError('cannot find root dir: {}'.format(root_dir))
        self.root_dir = root_dir
        
        self.img_files = [x for x in os.listdir(self.root_dir) if x.endswith('.png')]
        # for debugging
        # self.img_files = self.img_files[:20]
        self.data = []
        self.targets = []

        for img_file in self.img_files:
            img_path = os.path.join(self.root_dir, img_file)
            im = cv2.imread(img_path).astype(np.float32)
            
            im /= 255.0
            im -= mean
            im /= std
            # to tensor, H x W x C -> C x H x W
            im = torch.from_numpy(im).float().permute(2, 0, 1)

            self.data.append(im)

            name = os.path.splitext(img_path)[0]

            label_seq = name.split('-')[-1]

            label_seq = [int(x)+1 for x in label_seq]
            
            self.targets.append(torch.IntTensor(label_seq))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.targets[ind]

if __name__ == '__main__':
    from utils import DATASET_PATH
    dataset = CaptchaDataset(os.path.join(DATASET_PATH, 'train'), 
                             mean=[0., 0., 0.], std=[1.,1.,1.])
    print('total number: {}'.format(len(dataset)))

    data = torch.stack(dataset.data, dim=1).view((3, -1))
    # now data is 3 x (N x H x W)
    mean = torch.mean(data, dim=1)
    std = torch.std(data, dim=1)

    print ('mean = {},\nstd = {}'.format(mean, std))

    import random
    sample_inds = random.sample(range(0, len(dataset)), 4)
    samples = [dataset[i] for i in sample_inds]

    import matplotlib.pyplot as plt
    plt.figure()

    import itertools
    for i, (tensor_data, target) in enumerate(samples):
        img_array = tensor_data.permute(1, 2, 0).numpy()
        img_array = (img_array * 255).astype(np.uint8)
        plt.subplot(2, 2, i+1)
        plt.title(''.join(str(x-1) for  x in target))
        plt.imshow(img_array)

    plt.show()




        

