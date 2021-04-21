import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
import os
import logging
from utils.utils import *
from utils.compute_flops import lookup_table_flops
from utils.transfer_archs import decode_cfg
import torch.distributed as dist
from utils.dist_utils import *
import time
import pickle
from pdb import set_trace as br

class Architect(nn.Module):
    def __init__(self, model, args):
        super(Architect, self).__init__()
        self.args = args
        self.model = model
        # flops table loaded inside the learner
        self.optimizer = torch.optim.Adam(list(self.model.arch_parameters()),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.baseline = torch.tensor(0.).cuda()
        self.gamma = args.gamma

    def update_baseline(self, reward_raw):
        self.baseline = self.baseline * self.gamma + reward_raw * (1-self.gamma)

    def step(self, archs_logP, reduced_acc1, archs_entropy, arch_tmp):
        # NOTE: only update rl agent on rank 0
        policy_loss, reward_raw = self.model._loss_arch(archs_logP, reduced_acc1, archs_entropy, arch_tmp, self.baseline)

        if self.args.rank == 0:
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            self.update_baseline(reward_raw)

        if self.args.distributed:
            # sync baseline and arch master
            dist.barrier()
            dist.broadcast(self.baseline, 0)
            broadcast_params(self.model.arch_master)
            # check passed. params are the same on multiple GPU

        return reward_raw


class ChannIlsvrcLearner(object):
    def __init__(self, model, loaders, args, device):
        self.args = args
        self.device = device
        self.model = model
        self.proj_lr = 0. # initially do not change P,Q
        self.__build_path()
        self.train_loader, self.test_loader = loaders
        # self.writer = SummaryWriter(os.path.dirname(self.save_path))
        self.__build_learner()

    def __build_learner(self):
        # split variables to weights and arch_params
        self.__setup_optim()
        self.architect = Architect(self.model, self.args)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def train(self, samplers):
        train_sampler = samplers
        self.model.arch_master.force_uniform = True # initially random sample for warmup

        for epoch in range(self.args.epochs):
            if self.args.distributed:
                assert train_sampler is not None
                train_sampler.set_epoch(epoch)

            if epoch > self.args.warmup_epochs:
                self.model.arch_master.force_uniform = False
                if self.args.ft_schedual == 'follow_meta':
                    self.proj_opt.param_groups[0]['lr'] = self.proj_opt.param_groups[0]['lr']   # 0.001
                elif self.args.ft_schedual == 'fixed':
                    self.proj_opt.param_groups[0]['lr'] = self.args.ft_proj_lr
                else:
                    raise ValueError('Wrong Projection Fintuning Type!.')

            self.model.train()
            if self.check_is_primary():
                logging.info("Training at Epoch: %d" % epoch)
            train_acc, train_loss = self.epoch_train(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # in ilsvrc learner we warmup in advance. and (epoch > self.args.warmup_epochs):
            if (epoch + 1) % self.args.eval_epoch == 0:
                if self.check_is_primary():
                    self.save_model()
                    logging.info("Evaluation at Epoch: %d" % epoch)

                    if (epoch + 1) == self.args.epochs//2 and self.args.warmup_epochs == self.args.epochs:
                        # NOTE: store a 0.1 lr model separately
                        self.save_model('model_0.1.pt')
                        logging.info("Init lr model saved")
                if self.args.distributed:
                    dist.barrier()
                self.evaluate(True, epoch)

    def finetune(self, samplers):
        train_sampler = samplers
        self.load_model()
        self.evaluate(True, 0)
        for epoch in range(self.args.epochs):
            if self.args.distributed:
                assert train_sampler is not None
                train_sampler.set_epoch(epoch)

            if epoch > self.args.warmup_epochs:
                self.model.arch_master.force_uniform = False
                if self.args.ft_schedual == 'follow_meta':
                    self.proj_opt.param_groups[0]['lr'] = self.proj_opt.param_groups[0]['lr']   # 0.001
                elif self.args.ft_schedual == 'fixed':
                    self.proj_opt.param_groups[0]['lr'] = self.args.ft_proj_lr
                else:
                    raise ValueError('Wrong Projection Fintuning Type!.')

            self.model.train()
            if self.check_is_primary():
                logging.info("Finetuning at Epoch: %d" % epoch)
            ft_acc, ft_loss = self.epoch_train(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if (epoch + 1) % self.args.eval_epoch == 0:
                if self.check_is_primary():
                    self.save_model()
                    logging.info("Evaluation at Epoch: %d" % epoch)
                if self.args.distributed:
                    dist.barrier()
                self.evaluate(True, epoch)

    def evaluate(self, is_train=False, epoch=None):
        self.model.eval()
        if self.args.distributed:
            sync_bn_stat(self.model, self.args.world_size)

        if not is_train:
            self.load_model()

        with torch.no_grad():
            if self.args.beam_search:
                self.beam_search_eval()
                # self.epoch_eval(epoch)
            else:
                self.epoch_eval(epoch)

    def misc(self):
        # check ilsvrc data
        total_idx = len(self.train_loader)
        for idx, (data_train, data_valid) in enumerate(zip(self.train_loader, self.valid_loader)):
            input_x, target_y = data_train[0].to(self.device), data_train[1].to(self.device)
            input_search, target_search = data_valid[0].to(self.device), data_valid[1].to(self.device)
            if idx % 100 == 0:
                print("Reading... %.2f complete" % float(idx/total_idx))
        print("All input passed")

    def epoch_train(self, epoch):
        """ Rewrite this function if necessary in the sub-classes. """
        # setup statistics
        batch_time = AverageMeter('Time', ':3.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        lr = AverageMeter('Lr', ':.3e')
        losses = AverageMeter('Loss', ':.4e')
        losses_ce = AverageMeter('Loss_ce', ':.4e')
        losses_proj = AverageMeter('Loss_proj', ':.4e')

        # penalty = AverageMeter('Penalty', ':.4e')
        # flops = AverageMeter('Decode FLOPs', ':.4e')
        rewards = AverageMeter('Controller reward', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        top1 = AverageMeter('Acc@1', ':3.3f')
        top5 = AverageMeter('Acc@5', ':3.3f')
        metrics = [lr, batch_time, top1, top5, losses, losses_ce, losses_proj, rewards, entropy]

        loader_len = len(self.train_loader)
        progress = ProgressMeter(loader_len, *metrics, prefix='Job id: %s, ' % self.args.job_id)
        end = time.time()

        for idx, data_train in enumerate(self.train_loader):
            input_x, target_y = data_train[0].to(self.device), data_train[1].to(self.device)
            logits, archs_logP, archs_entropy, arch_tmp = self.model(input_x)
            # check passed: archs are all the same across multiple GPUS.
            # NOTE: archs_entropy and logP may be different, but we only update
            # arch_master on rank==0.

            loss_ce = self.criterion(logits, target_y) / self.args.world_size

            acc1, acc5 = accuracy(logits, target_y, topk=(1, 5))

            reduced_loss_ce = loss_ce.data.clone()
            reduced_acc1 = acc1.clone() / self.args.world_size
            reduced_acc5 = acc5.clone() / self.args.world_size

            if self.args.distributed:
                dist.all_reduce(reduced_loss_ce)
                dist.all_reduce(reduced_acc1)
                dist.all_reduce(reduced_acc5)

            # NOTE: policy_loss and loss_ce, loss_proj  are w.r.t different graphs.
            # Therefore freeing graph for policy_loss does not affect loss_ce
            # and loss_proj`

            # update alpha on validation set after warmup
            if epoch > self.args.warmup_epochs:
                reward_raw = self.architect.step(archs_logP, reduced_acc1, archs_entropy, arch_tmp)
                rewards.update(reward_raw.item(), n=1)
                entropy.update(archs_entropy.item(), n=1)

            # update meta and projection weights
            self.opt.zero_grad()
            loss_ce.backward()
            if self.args.distributed:
                average_group_gradients(self.model.meta_parameters())
            self.opt.step()

            if idx % self.args.updt_proj == 0:
                # NOTE: now we update orthogonal loss seperately inside
                loss_proj = self.model.updt_orthogonal_reg_loss(self.proj_opt) # return a python scalar

                # project back to unit lenght after warmup
                if self.args.norm_constraint == 'constraint' and self.proj_opt.param_groups[0]['lr'] > 0:
                    for k, v in self.model.named_projection_parameters():
                        v_sum = v.transpose(1, 0).mm(v).sqrt().diag()
                        v_sum = v_sum.repeat(v.size(0), 1)
                        v.data = (v / v_sum).data
                elif self.args.norm_constraint != 'constraint':
                    raise ValueError

            # update statistics
            top1.update(reduced_acc1[0].item(), input_x.shape[0])
            top5.update(reduced_acc5[0].item(), input_x.shape[0])
            losses.update(loss_proj+reduced_loss_ce.item(), input_x.shape[0])
            losses_ce.update(reduced_loss_ce.item(), input_x.shape[0])
            losses_proj.update(loss_proj, input_x.shape[0])
            lr.update(self.opt.param_groups[0]['lr'])

            batch_time.update(time.time() - end)
            end = time.time()

            # show the training/evaluating statistics
            if self.check_is_primary() and ((idx % self.args.print_freq == 0) or (idx + 1) % loader_len == 0):
                progress.show(idx)

        return top1.avg, losses.avg

    def beam_search_eval(self, epoch=None):
        best_acc = -np.inf
        # best_loss = np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_arch = None
        best_logits = None
        beam_size = self.args.top_seq
        cand_seq, logits_seq, logP_accum, entropy_accum = self.model.arch_master.beam_search(beam_size)
        if self.args.distributed:
            dist.broadcast(cand_seq, 0)
            # NOTE: archs_seq are the same, no noed to broadcase
            # print(self.args.rank, logits_seq[0])
            # dist.broadcast(logits_seq, 0)
            dist.broadcast(logP_accum, 0)
            dist.broadcast(entropy_accum, 0)

        parallel_eval = True if self.args.distributed and beam_size%self.args.world_size==0 and not self.args.fix_random \
            else False

        if parallel_eval:
            idx = 0
            while idx < beam_size:
                top1 = AverageMeter('cand top1', ':3.3f')
                arch_id = idx + self.args.rank
                cand = cand_seq[arch_id]
                # arch = [self.model.candidate_width[v] for v in cand]
                # print("On rank: %d, Evaluating the %d-th arch, archs: %s" % (self.args.rank, arch_id, str(cand)))
                print("On rank: %d, %d-th Arch: %s" % (self.args.rank, arch_id, \
                    decode_cfg(self.args, cand, self.model.num_blocks, self.model.block_layer_num)))

                # NOTE: comment this for fast eval
                for test_input, test_target in self.test_loader:
                    test_input, test_target = test_input.to(self.device), test_target.to(self.device)
                    logits = self.model.test_forward(test_input, cand)
                    acc = accuracy(logits, test_target)[0]
                    top1.update(acc.item(), test_input.size(0))

                flops = self.model.arch_master._compute_flops(cand)
                # print all the sampled archs in parallel
                print("Rank: %d, Arch id:%d, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                    (self.args.rank, arch_id, top1.avg, logP_accum[arch_id].item(), entropy_accum[arch_id].item(), flops))

                # init group vars to be gathered
                top1 = torch.tensor(top1.avg).float().cuda()
                g_top1 = [torch.ones_like(top1) for _ in range(self.args.world_size)]
                # collect results on different GPUs
                dist.all_gather(g_top1, top1)
                if self.check_is_primary():
                    max_ind = np.argmax(g_top1)
                    if g_top1[max_ind] > best_acc:
                        best_acc = g_top1[max_ind]
                        best_arch = cand
                        best_arch_logP = logP_accum[idx+max_ind].item()
                        best_arch_ent = entropy_accum[idx+max_ind].item()
                idx += self.args.world_size

        else:
            if self.check_is_primary():
                for idx, cand in enumerate(cand_seq):
                    # enumerate over each cand arch and perform testing
                    top1 = AverageMeter('cand top1', ':3.3f')

                    # NOTE: comment this for fast eval
                    for test_input, test_target in self.test_loader:
                        test_input, test_target = test_input.to(self.device), test_target.to(self.device)
                        logits = self.model.test_forward(test_input, cand)
                        acc = accuracy(logits, test_target)[0]
                        top1.update(acc.item(), test_input.size(0))

                    flops = self.model.arch_master._compute_flops(cand)

                    # print all the sampled archs on primary rank
                    print("%d-th Arch: %s" % (idx, decode_cfg(self.args, cand, self.model.num_blocks, self.model.block_layer_num)))
                    print("Arch id:%d, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                        (idx, top1.avg, logP_accum[idx].item(), entropy_accum[idx].item(), flops))

                    if top1.avg > best_acc:
                        best_acc = top1.avg
                        best_arch = cand
                        best_arch_logP = logP_accum[idx].item()
                        best_arch_ent = entropy_accum[idx].item()

        if self.check_is_primary() and self.model.num_cand> 1:
            avg_logits = [torch.stack(logits) for logits in logits_seq]
            avg_logits = torch.stack(avg_logits).mean(0)
            avg_arch_info, avg_discrepancy = self.model.get_arch_info(avg_logits)
            print(avg_arch_info)
            logging.info("Best: Accuracy %f -LogP %f ENT %f",best_acc, -best_arch_logP, best_arch_ent)
            logging.info("Best Arch: %s" % str(best_arch))
            logging.info("Beam search done. size: %d" % beam_size)

        # sync back
        if self.args.distributed:
            dist.barrier()

    def epoch_eval(self, epoch):
        best_acc = -np.inf
        # best_loss = np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_arch = None
        avg_logits_list = []

        parallel_eval = True if self.args.distributed and \
            self.args.n_test_archs % self.args.world_size == 0 and not self.args.fix_random \
            else False

        if parallel_eval:
            # NOTE: a new parallel way to perform evaluation with differet arch
            if self.check_is_primary():
                logging.info("Now parallel evaluating different archs")

            idx = 0
            while (idx<self.args.n_test_archs):
                top1, arch_cand, arch_logP, arch_entropy, arch_info, discrepancy = self.model.test_cand_arch(self.test_loader)
                flops = self.model.arch_master._compute_flops(arch_cand)
                # logging.info("Rank:%d, Arch id:%d, %s, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                #     (self.args.rank, idx+self.args.rank, str(arch_cand.tolist()), top1, arch_logP.item(), arch_entropy.item(), flops))    # print all the sampled archs
                print("Rank:%d, Arch id:%d, %s, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                    (self.args.rank, idx+self.args.rank, str(arch_cand.tolist()), top1, arch_logP.item(), arch_entropy.item(), flops))    # print all the sampled archs

                dist.barrier()
                idx += self.args.world_size

                top1 = torch.tensor(top1).float().cuda()
                # init group vars to be gathered
                g_top1 = [torch.ones_like(top1) for _ in range(self.args.world_size)]
                g_logits = [torch.ones_like(self.model.logits) for _ in range(self.args.world_size)]
                g_arch_cand = [torch.ones_like(arch_cand) for _ in range(self.args.world_size)]
                g_entropy = [torch.ones_like(arch_entropy) for _ in range(self.args.world_size)]
                g_arch_logP = [torch.ones_like(arch_logP) for _ in range(self.args.world_size)]

                # collect results on different GPUs
                dist.all_gather(g_top1, top1)
                dist.all_gather(g_arch_logP, arch_logP)
                dist.all_gather(g_entropy, arch_entropy)
                dist.all_gather(g_arch_cand, arch_cand)
                dist.all_gather(g_logits, self.model.logits)

                if self.check_is_primary():
                    avg_logits_list += g_logits
                    max_ind = np.argmax(g_top1)
                    if g_top1[max_ind] > best_acc:
                       best_acc = g_top1[max_ind]
                       best_arch = g_arch_cand[max_ind]
                       best_arch_logP = g_arch_logP[max_ind]
                       best_arch_ent = g_entropy[max_ind]
                dist.barrier()

        else:
            if self.check_is_primary():
                # sample 20 archs and take the best one.
                logging.info("Single model evluating...")
                for i in range(self.args.n_test_archs):
                    top1, arch_cand, arch_logP, arch_entropy, arch_info, discrepancy = self.model.test_cand_arch(self.test_loader)
                    flops = self.model.arch_master._compute_flops(arch_cand)
                    logging.info("Arch: %s", decode_cfg(self.args, arch_cand, self.model.num_blocks, self.model.block_layer_num))
                    logging.info("Arch id:%d, %s, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                        (i, str(arch_cand.tolist()), top1, arch_logP.item(), arch_entropy.item(), flops))    # print all the sampled archs
                    avg_logits_list.append(self.model.logits)
                    if top1 > best_acc:
                        best_acc = top1
                        best_arch = arch_cand
                        best_arch_logP = arch_logP
                        best_arch_ent = arch_entropy

        if self.check_is_primary() and self.model.num_cand > 1:
            avg_logits = torch.stack(avg_logits_list)
            avg_arch_info, avg_discrepancy = self.model.get_arch_info(avg_logits.mean(0))
            print(avg_arch_info)
            logging.info("Best: Accuracy %f -LogP %f ENT %f",best_acc,
                             -best_arch_logP, best_arch_ent)
            logging.info("Best Arch: %s", decode_cfg(self.args, best_arch, self.model.num_blocks, self.model.block_layer_num))
            logging.info("Random sample evaluation done.")

        # sync back
        if self.args.distributed:
            dist.barrier()

    def __setup_optim(self):
        """ Set up optimizer for network parameters and projection matrix seperately (not arch parameters) """
        self.opt = optim.SGD(self.model.meta_parameters(), lr=self.args.lr, momentum=self.args.momentum, \
            nesterov=self.args.nesterov, weight_decay=self.args.weight_decay)

        self.proj_opt = optim.SGD(self.model.projection_parameters(), lr=self.proj_lr, momentum=self.args.momentum, \
            nesterov=self.args.nesterov,  weight_decay=self.args.weight_decay)

        # proj_lr is adjusted in self.train()
        if self.args.lr_decy_type == 'multi_step':
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[\
                int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)])
        elif self.args.lr_decy_type == 'cosine':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(\
                self.opt, self.args.epochs, eta_min=self.args.lr_min)
        else:
            raise ValueError("Unknown model, failed to initalize optim")

    def __build_path(self):
        if self.args.exec_mode == 'train':
            self.save_path = os.path.join(self.args.save_path,
                                          '_'.join([self.args.model_type, self.args.learner]),
                                          self.args.job_id, 'model.pt')
            self.load_path = self.save_path
        elif self.args.exec_mode == 'finetune':
            self.load_path = self.args.load_path
            if self.args.warmup_epochs == self.args.epochs:
                # further warmup with decayed learning rate
                self.save_path = os.path.join(os.path.dirname(self.load_path), 'model_ft_%s.pt' % self.args.job_id)
            else:
                self.save_path = os.path.join(os.path.dirname(self.load_path), 'model_search_%s.pt' % self.args.job_id)
        else:
            self.load_path = self.args.load_path
            self.save_path = self.load_path

    def check_is_primary(self):
        if (self.args.distributed and self.args.rank == 0) or \
            not self.args.distributed:
            return True
        else:
            return False

    def save_model(self, file_name=None):
        if file_name is None:
            file_name = self.save_path
        else:
            file_name = os.path.join(os.path.dirname(self.save_path), file_name)

        state = {'state_dict': self.model.state_dict(), \
            'optimizer': self.opt.state_dict(), \
            'arch_optimizer': self.architect.optimizer.state_dict()}
        torch.save(state, file_name)
        logging.info("Model stored at: " + file_name)

    def load_model(self):
        if self.args.distributed:
            # read parameters to each GPU seperately
            loc = 'cuda:{}'.format(torch.cuda.current_device())
            checkpoint = torch.load(self.load_path, map_location=loc)
        else:
            checkpoint = torch.load(self.load_path)

        self.model.load_state_dict(checkpoint['state_dict'])
        # NOTE: for wamrup, useless to restore optimizer params.
        # self.opt.load_state_dict(checkpoint['optimizer'])
        # self.architect.optimizer.load_state_dict(checkpoint['arch_optimizer'])
        logging.info("Model succesfully restored from %s" % self.load_path)

        if self.args.distributed:
            broadcast_params(self.model)

