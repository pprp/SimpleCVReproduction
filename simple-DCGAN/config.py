class Config(object):
    data_path = "./data/faces"

    num_workers = 2
    image_size = 96
    batch_size = 64
    max_epoch = 300

    lr1 = 2e-4
    lr2 = 2e-4

    beta1 = 0.5

    use_gpu = True

    nz = 100
    ngf = 64
    ndf = 64

    save_path = './imgs/'
    vis = True
    env = 'dcgan'
    plot_every = 20

    debug_file = '/tmp/debuggan'

    d_every = 1
    g_every = 5
    save_every = 10

    netd_path = None
    netg_path = None

    # testing
    gen_img = 'result.png'

    gen_num = 64
    gen_search_num = 512
    gen_mean = 0
    gen_std = 1


opt = Config()
