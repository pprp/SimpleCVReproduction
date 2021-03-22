# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from models.normal_nets.proxyless_nets import ProxylessNASNets
from run_manager import RunConfig


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    else:
        raise ValueError('unrecognized type of network: %s' % name)


class ImagenetRunConfig(RunConfig):

    def __init__(self, train_dir='', test_dir='', n_epochs=150, init_lr=0.05, gpu=0, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=256, test_batch_size=500, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.1, no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10, train_iters=-1,
                 n_worker=32, resize_scale=0.08, distort_color='normal',  **kwargs):
        super(ImagenetRunConfig, self).__init__(
            train_dir, test_dir, n_epochs, init_lr, gpu, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency, train_iters
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.gpu = gpu
        self.train_dir = train_dir
        self.test_dir = test_dir

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
            'resize_scale': self.resize_scale,
            'distort_color': self.distort_color,
        }

