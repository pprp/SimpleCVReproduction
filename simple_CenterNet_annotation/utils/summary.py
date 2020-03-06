import os
import sys
import torch
import logging
from datetime import datetime

# return a fake summarywriter if tensorbaordX is not installed

try:
    from tensorboardX import SummaryWriter
except ImportError:

    class SummaryWriter:
        def __init__(self, log_dir=None, comment='', **kwargs):
            print(
                '\nunable to import tensorboardX, log will be recorded by pytorch!\n'
            )
            self.log_dir = log_dir if log_dir is not None else './logs'
            os.makedirs('./logs', exist_ok=True)
            self.logs = {'comment': comment}
            return

        def add_scalar(self,
                       tag,
                       scalar_value,
                       global_step=None,
                       walltime=None):
            if tag in self.logs:
                self.logs[tag].append((scalar_value, global_step, walltime))
            else:
                self.logs[tag] = [(scalar_value, global_step, walltime)]
            return

        def close(self):
            timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_')
            torch.save(self.logs,
                       os.path.join(self.log_dir, 'log_%s.pickle' % timestamp))
            return


class EmptySummaryWriter:
    def __init__(self, **kwargs):
        pass

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        pass

    def close(self):
        pass


def create_summary(distributed_rank=0, **kwargs):
    if distributed_rank > 0:
        return EmptySummaryWriter(**kwargs)
    else:
        return SummaryWriter(**kwargs)


def create_logger(distributed_rank=0, save_dir=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    filename = "log_%s.txt" % (datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s [%(asctime)s]")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Saver:
    def __init__(self, distributed_rank, save_dir):
        self.distributed_rank = distributed_rank
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        return

    def save(self, obj, save_name):
        if self.distributed_rank == 0:
            torch.save(obj, os.path.join(self.save_dir, save_name + '.t7'))
            return 'checkpoint saved in %s !' % os.path.join(
                self.save_dir, save_name)
        else:
            return ''


def create_saver(distributed_rank, save_dir):
    return Saver(distributed_rank, save_dir)


class DisablePrint:
    def __init__(self, local_rank=0):
        self.local_rank = local_rank

    def __enter__(self):
        if self.local_rank != 0:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        else:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.local_rank != 0:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        else:
            pass


if __name__ == '__main__':
    sw = SummaryWriter()
    sw.close()
