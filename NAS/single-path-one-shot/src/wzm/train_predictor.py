import sys
import torch
import numpy as np
import os
from torch import optim
from mlp_predictor import AccuracyPredictor
from arch_dataset import build_dataloader, load_from_json
from coder import GeneCoder
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import stats
import utils
import torch.nn.functional as F
import time

logger = utils.get_logger()

writer = SummaryWriter("Logs/runs")

# class Logger():
#     rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
#     log_path = './Logs/'
#     log_name = log_path + rq + '.log'
#     logfile = log_name
#     if not os.path.exists(logfile):
#         os.system(r"touch {}".format(logfile))  # 调用系统命令行来创建文件
#     def __init__(self, filename=logfile):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass
#
# sys.stdout = Logger()


def train_pred(num_epoches, model, train_loader, val_loader, optimizer):
    low=100
    for epoch in range(num_epoches):
        train_loss = train_epoch(logger,train_loader, model, epoch, optimizer)
        logger.info("Train: Epoch {:3d}: train loss {:.4f}".format(epoch, train_loss))
        val_loss = valid_epoch(val_loader, model,epoch)
        logger.info("Valid: Epoch {:3d}: val loss {:.4f}".format(epoch, val_loss))
        if val_loss<low:
            low=val_loss
            print('Saving ..')
            torch.save(model.state_dict(), os.path.join('predictor_checkpoint', 'mlp.pth'))
        print('\n')
    logger.info("Best rmse loss {:.4f} ".format(low))


def valid_epoch(val_loader, model, epoch):
    val_loss = utils.AverageMeter()
    model.eval()
    with torch.no_grad():
        for step, (archs, accs) in enumerate(val_loader):
            n = len(archs)
            scores = model.predict(archs)
            rmse_loss = torch.sqrt(F.mse_loss(scores.squeeze(), scores.new(accs)))
            writer.add_scalar("valid_loss", rmse_loss, epoch)
            val_loss.update(rmse_loss.item(), n)
            # if step % 10 == 0:
            #     logger.info("valid {:03d} [{:03d}/{:03d}] {:.4f}".format(
            #         epoch, step, len(val_loader), val_loss.avg))
    return val_loss.avg


def train_epoch(logger, train_loader, model, epoch, optimizer):
    train_loss = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    model.train()
    for step, (archs, accs) in enumerate(train_loader):
        n = len(archs)
        n_max_pairs = int(4 * n)
        acc_diff = np.array(accs)[:, None] - np.array(accs)
        acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_diff_matrix > 0)
        ex_thresh_num = len(ex_thresh_inds[0])
        if ex_thresh_num > n_max_pairs:
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])
        archs_1, archs_2, better_lst = archs[ex_thresh_inds[1]], archs[ex_thresh_inds[0]], \
                                       (acc_diff > 0)[ex_thresh_inds]
        n_diff_pairs = len(better_lst)
        n_diff_pairs_meter.update(float(n_diff_pairs))
        s_1 = model.predict(archs_1)
        s_2 = model.predict(archs_2)
        better_pm = 2 * s_1.new(np.array(better_lst, dtype=np.float32)) - 1
        zero_ = s_1.new([0.])
        margin = [0.01]
        margin = s_1.new(margin)
        pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        scores =model.predict(archs)
        rmse_loss = torch.sqrt(F.mse_loss(scores.squeeze(), scores.new(accs)))   # RMSE loss
        # print('pair loss:',  pair_loss)
        # print('rmse loss:', rmse_loss)
        added_loss = 0.2*pair_loss + 0.8* rmse_loss
        writer.add_scalar("added_loss", added_loss, epoch)
        optimizer.zero_grad()
        added_loss.backward()
        optimizer.step()
        train_loss.update(added_loss.item(), n)
        # if step % 10 == 0:
        #     logger.info("train {:03d} [{:03d}/{:03d}] {:.4f}".format(
        #         epoch, step, len(train_loader), train_loss.avg))
    return train_loss.avg



if __name__ == '__main__':
    ss_path = "search_space.json"
    coder_ = GeneCoder(ss_path)
    arch_data = load_from_json('Track2_final_archs.json')
    files_path = 'generation_checkpoint/2.pth'

    gen_checkpoint = torch.load((files_path), map_location=torch.device('cpu'))
    pred = gen_checkpoint['pred']
    arch = gen_checkpoint['arch']
    arch_data.extend(pred)
    for p in pred:
        for i in range(5):
            trans_param=coder_.mutate_one_value(p[:-1])
            arch_data.append(trans_param + [p[-1]])


    # print(arch_data)
    model = AccuracyPredictor()
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    train_loader, val_loader = build_dataloader(arch_data, batch_size=16, coder=coder_)
    train_pred(30, model, train_loader, val_loader, optimizer)


