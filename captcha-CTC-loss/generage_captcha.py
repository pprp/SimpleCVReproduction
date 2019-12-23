from captcha.image import ImageCaptcha
import os
import numpy as np
from multiprocessing import Process
from utils import DATASET_PATH
LABEL_SEQ_LENGTH = 5
BLANK_LABEL = 10

threads_num = 32
img_path = DATASET_PATH

def generate_random_label(length):
    """
    generate labels, we use 10 as blank
    """
    not_blank = []
    while len(not_blank) == 0:
        rand_array = np.random.randint(11, size=length)
        not_blank = rand_array[rand_array != BLANK_LABEL]

    return ''.join(map(lambda x: str(x), not_blank))

image = ImageCaptcha()

def generate_image(seed, path, start, end):
    np.random.seed(seed)
    for idx in range(start, end):
        length = np.random.randint(LABEL_SEQ_LENGTH-1) + 1
        label_seq = generate_random_label(length)
        image.write(label_seq, os.path.join(path, '%05d-'%idx + label_seq + '.png'))

def get_batchsize(total_num, groups):
    bsz = total_num // groups
    if total_num % groups:
        bsz += 1
    return bsz

for phase in ['test', 'train']:
    path = os.path.join(img_path, phase)
    if not os.path.exists(path):
        os.mkdir(path)
    if phase == 'test':
        DATASET_SIZE = 10000
    else:
        DATASET_SIZE = 50000
    threads = []
    batch_size = get_batchsize(DATASET_SIZE, threads_num)

    for t in range(threads_num):
        start, end = t*batch_size, (t+1)*batch_size
        if t == threads_num - 1:
            end = DATASET_SIZE
        p = Process(target = generate_image, args = (t, path, start, end))
        p.start()
        threads.append(p)

    for p in threads:
        p.join()
    print(phase + ' done.')
    