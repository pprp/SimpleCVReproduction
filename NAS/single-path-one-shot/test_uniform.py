import numpy as np

flops_l, flops_r, flops_step = 290, 360, 10
bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]

def get_random_cand():
    return tuple(np.random.randint(4) for i in range(20))

def get_uniform_sample_cand(*, timeout=500):
    idx = np.random.randint(len(bins)) # 7个bin，7个layer
    for i in range(timeout):
        cand = get_random_cand()
        return cand
    return get_random_cand()

print(bins)
print(get_uniform_sample_cand(timeout=20))
