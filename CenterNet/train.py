from utils.utils import _tranpose_and_gather_feature, load_model
from utils.summary import (DisablePrint, create_logger, create_saver,
                           create_summary)
from utils.post_process import ctdet_decode
from utils.losses import _neg_loss, _reg_loss
from utils.image import transform_preds
from nets.resdcn import get_pose_net
from nets.hourglass import get_hourglass
from datasets.pascal import PascalVOC, PascalVOC_eval
from datasets.coco import COCO, COCO_eval
import argparse
import os
import sys
import time

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str,
                    default='pascal_small_hourglass_512_dp')
parser.add_argument('--pretrain_name', type=str, default='pretrain')

parser.add_argument('--dataset',
                    type=str,
                    default='pascal',
                    choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='small_hourglass')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=4)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)
cfg.pretrain_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.pretrain_name,
                                'checkpoint.t7')

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print = logger.info
    print(cfg)

    torch.manual_seed(317)
    # disable this if OOM at beginning of training
    torch.backends.cudnn.benchmark = True

    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=num_gpus,
                                rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    print('Setting up data...')
    Dataset = COCO if cfg.dataset == 'coco' else PascalVOC

    train_dataset = Dataset(cfg.data_dir,
                            'train',
                            split_ratio=cfg.split_ratio,
                            img_size=cfg.img_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_gpus, rank=cfg.local_rank)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // num_gpus if cfg.dist else cfg.batch_size,
        shuffle=not cfg.dist,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler if cfg.dist else None)

    Dataset_eval = COCO_eval if cfg.dataset == 'coco' else PascalVOC_eval

    val_dataset = Dataset_eval(cfg.data_dir,
                               'val',
                               test_scales=[1.],
                               test_flip=False)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=1,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    print('Creating model...')
    if 'hourglass' in cfg.arch:
        model = get_hourglass[cfg.arch]
    elif 'resdcn' in cfg.arch:
        model = get_pose_net(num_layers=int(cfg.arch.split('_')[-1]),
                             num_classes=train_dataset.num_classes)
    else:
        raise NotImplementedError

    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[
                cfg.local_rank,
            ], output_device=cfg.local_rank)
    else:
        model = nn.DataParallel(model).to(cfg.device)

    if os.path.isfile(cfg.pretrain_dir):
        model = load_model(model, cfg.pretrain_dir)

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        cfg.lr_step,
                                                        gamma=0.1)

    def train(epoch):
        print('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=cfg.device,
                                           non_blocking=True)

            outputs = model(batch['image'])

            # 得到heat map, reg, wh 三个变量
            hmap, regs, w_h_ = zip(*outputs)

            regs = [
                _tranpose_and_gather_feature(r, batch['inds']) for r in regs
            ]
            w_h_ = [
                _tranpose_and_gather_feature(r, batch['inds']) for r in w_h_
            ]

            # 分别计算loss
            hmap_loss = _neg_loss(hmap, batch['hmap'])
            reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])

            # 进行loss加权，得到最终loss
            loss = hmap_loss + 1 * reg_loss + 0.1 * w_h_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print('[%d/%d-%d/%d] ' %
                      (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                      ' hmap_loss= %.5f reg_loss= %.5f w_h_loss= %.5f' %
                      (hmap_loss.item(), reg_loss.item(), w_h_loss.item()) +
                      ' (%d samples/sec)' %
                      (cfg.batch_size * cfg.log_interval / duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
                summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
                summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
        return

    def val_map(epoch):
        print('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        max_per_image = 100

        results = {}
        with torch.no_grad():
            for inputs in val_loader:
                img_id, inputs = inputs[0]

                detections = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(
                        cfg.device)
                    output = model(inputs[scale]['image'])[-1]

                    dets = ctdet_decode(*output, K=cfg.test_topk)
                    dets = dets.detach().cpu().numpy().reshape(
                        1, -1, dets.shape[2])[0]

                    top_preds = {}
                    dets[:, :2] = transform_preds(
                        dets[:, 0:2], inputs[scale]['center'],
                        inputs[scale]['scale'],
                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    dets[:, 2:4] = transform_preds(
                        dets[:, 2:4], inputs[scale]['center'],
                        inputs[scale]['scale'],
                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    clses = dets[:, -1]
                    for j in range(val_dataset.num_classes):
                        inds = (clses == j)
                        top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                        top_preds[j + 1][:, :4] /= scale

                    detections.append(top_preds)

                bbox_and_scores = {
                    j: np.concatenate([d[j] for d in detections], axis=0)
                    for j in range(1, val_dataset.num_classes + 1)
                }
                scores = np.hstack([
                    bbox_and_scores[j][:, 4]
                    for j in range(1, val_dataset.num_classes + 1)
                ])
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, val_dataset.num_classes + 1):
                        keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                        bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

                results[img_id] = bbox_and_scores

        eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
        print(eval_results)
        summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)

    print('Starting training...')
    for epoch in range(1, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val_map(epoch)
        print(saver.save(model.module.state_dict(), 'checkpoint'))
        lr_scheduler.step(epoch)  # move to here after pytorch1.1.0

    summary_writer.close()


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
