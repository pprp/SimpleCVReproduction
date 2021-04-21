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
import time
import pickle
from pdb import set_trace as br


class Architect(nn.Module):
    def __init__(self, model, args):
        super(Architect, self).__init__()
        self.model = model

        if args.flops:
            with open(r"./flops_table_resnet20_thinner.pkl", "rb") as input_file:
                e = pickle.load(input_file)
        else:
            with open(r"./flops_table.pkl", "rb") as input_file:
                e = pickle.load(input_file)
        self.model.arch_master.table_flops = e
        self.optimizer = torch.optim.Adam(list(self.model.arch_parameters()),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.baseline = 0
        self.gamma = args.gamma

    def update_baseline(self, reward):
        self.baseline = self.baseline * self.gamma + reward * (1-self.gamma)


    def step(self, input_valid, target_valid):
        self.optimizer.zero_grad()
        loss, reward, entropy = self.model._loss_arch(input_valid, target_valid, self.baseline)  # compute flops of models generated here.
        loss.backward()
        self.optimizer.step()
        self.update_baseline(reward)
        return reward, entropy


class ChannCifarLearner(object):
    def __init__(self, model, loaders, args, device):
        self.args = args
        self.device = device
        self.model = model
        self.proj_lr = 0.0
        self.__build_path()
        self.train_loader, self.valid_loader, self.test_loader = loaders
        # self.writer = SummaryWriter(os.path.dirname(self.save_path))
        self.__build_learner()

    def __build_learner(self):
        # split variables to weights and arch_params
        self.model_parameters = [v for n, v in self.model.named_parameters() if 'aux' not in n]
        self.aux_parameeters = [v for n, v in self.model.named_parameters() if 'aux' in n]
        self.arch_parameters = list(self.model.arch_parameters())

        print('There are ', len(self.model_parameters), 'model parameters')
        print('There are ', len(self.aux_parameeters), 'aux parameters')
        print('There are ', len(self.arch_parameters), 'arch parameters')

        self.__setup_optim()
        self.architect = Architect(self.model, self.args)

    def train(self, samplers=None):
        self.model.arch_master.force_uniform = True # initially uniform sampling for warmup
        for epoch in range(self.args.epochs):
            self.model.train()

            if self.lr_scheduler:
                self.lr_scheduler.step()
                self.aux_lr_scheduler.step()

            logging.info("Training at Epoch: %d" % epoch)
            train_acc, train_loss = self.epoch_train(epoch)

            if (epoch + 1) % self.args.eval_epoch == 0:
                logging.info("Evaluation at Epoch: %d" % epoch)
                self.evaluate(True, epoch)

                # save the model
                torch.save({"model":self.model.state_dict(), "arch_parameters":self.arch_parameters}, self.save_path)
                logging.info("Model stored at: " + self.save_path)

                # save the last warmup model
                if epoch == self.args.warmup_epochs:
                    warmup_save_path = os.path.join(os.path.dirname(self.save_path), 'model_warmup.pt')
                    logging.info("Warmup done. Warmup model stored at:" + warmup_save_path)

    def evaluate(self, is_train=False, epoch=None):
        self.model.eval()
        if not is_train:
            self.model.load_state_dict(torch.load(self.load_path)["model"])
            self.model.arch_master.force_uniform = False
            logging.info("Model successfully restored from %s" % self.load_path)

        with torch.no_grad():
            if self.args.beam_search:
                self.beam_search_eval()
            else:
                self.epoch_eval(epoch)

    def finetune(self, samplers=None):
        self.model.load_state_dict(torch.load(self.load_path)["model"])
        self.model.arch_master.force_uniform = False
        logging.info("Model successfully restored from %s" % self.load_path)

        # self.evaluate(True, 0)
        for epoch in range(self.args.epochs):

            self.model.train()
            logging.info("Finetuning at Epoch: %d" % epoch)

            ft_acc, ft_loss = self.epoch_train(epoch)

            if self.lr_scheduler:
                self.lr_scheduler.step()
                self.aux_lr_scheduler.step()

            if (epoch + 1) % self.args.eval_epoch == 0:
                logging.info("Evaluation at Epoch: %d" % epoch)
                self.evaluate(True, epoch)

                # save the model
                torch.save({"model":self.model.state_dict(), "arch_parameters":self.arch_parameters}, self.save_path)
                logging.info("Model stored at: " + self.save_path)

    def epoch_train(self, epoch):
        """ Rewrite this function if necessary in the sub-classes. """

        # setup statistics
        batch_time = AverageMeter('Time', ':3.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        lr = AverageMeter('Lr', ':.3e')
        losses = AverageMeter('Loss', ':.4e')
        losses_orthg = AverageMeter('Loss_orthg', ':.4e')
        losses_norm = AverageMeter('Loss_norm', ':.4e')
        # penalty = AverageMeter('Penalty', ':.4e')
        # flops = AverageMeter('Decode FLOPs', ':.4e')
        rewards = AverageMeter('Controller reward', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        top1 = AverageMeter('Acc@1', ':3.3f')
        top5 = AverageMeter('Acc@5', ':3.3f')
        metrics = [lr, batch_time, top1, top5, losses, losses_orthg, losses_norm, rewards, entropy]

        loader_len = len(self.train_loader)
        progress = ProgressMeter(loader_len, *metrics, prefix='Job id: %s, ' % self.args.job_id)
        end = time.time()

        for idx, (data_train, data_valid) in enumerate(zip(self.train_loader, self.valid_loader)):
            input_x, target_y = data_train[0].to(self.device), data_train[1].to(self.device)
            input_search, target_search = data_valid[0].to(self.device), data_valid[1].to(self.device)
            # update alpha on validation set after warmup
            if epoch > self.args.warmup_epochs:
                if self.args.ft_schedual == 'follow_meta':
                    self.opt.param_groups[1]['lr'] = self.opt.param_groups[0]['lr']   # 0.001
                elif self.args.ft_schedual == 'fixed':
                    self.opt.param_groups[1]['lr'] = self.args.ft_proj_lr
                else:
                    raise ValueError('Wrong Projection Fintuning Type!.')
                self.model.arch_master.force_uniform = False
                reward, ent = self.architect.step(input_search, target_search)
                rewards.update(reward.item(), n=1)
                entropy.update(ent.item(), n=1)

            logits, aux_logits_list, _, _, _ = self.model(input_x)
            if idx % self.args.updt_proj == 0:
                loss_orthognal, loss_norm = self.model.orthogonal_reg_loss()
            else:
                loss_orthognal, loss_norm = loss_orthognal.data.clone(), loss_norm.data.clone()

            # update aux classifers
            aux_losses = 0
            if self.args.use_aux:
                for aux_logits in aux_logits_list:
                    aux_loss = nn.CrossEntropyLoss()(aux_logits, target_y)
                    aux_losses += aux_loss
                self.aux_opt.zero_grad()
                aux_losses.backward()
                self.aux_opt.step()

            loss = nn.CrossEntropyLoss()(logits, target_y)
            # updt main loss
            if self.args.norm_constraint == 'regularization':
                loss += self.args.orthg_weight * loss_orthognal + self.args.norm_weight * loss_norm

            elif self.args.norm_constraint == 'constraint':
                loss += self.args.orthg_weight * loss_orthognal
                # loss = self.args.orthg_weight * loss_orthognal
            else:
                raise ValueError('Wrong norm constraint!')

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            if self.args.norm_constraint == 'constraint':
                # add projected optimization of projection matrices.
                for k, v in self.model.named_projection_parameters():
                    v_sum = v.transpose(1, 0).mm(v).sqrt().diag()
                    v_sum = v_sum.repeat(v.size(0), 1)
                    v.data = (v / v_sum).data

            # calculate accs
            acc1, acc5 = accuracy(logits, target_y, topk=(1, 5))

            top1.update(acc1[0], input_x.shape[0])
            top5.update(acc5[0], input_x.shape[0])
            losses.update(loss.item(), input_x.shape[0])
            losses_orthg.update(loss_orthognal.item(), input_x.shape[0])
            losses_norm.update(loss_norm.item(), input_x.shape[0])
            lr.update(self.opt.param_groups[0]['lr'])

            batch_time.update(time.time() - end)
            end = time.time()

            # show the training/evaluating statistics
            if (idx % self.args.print_freq == 0) or (idx + 1) % loader_len == 0:
                progress.show(idx)

        return top1.avg, losses.avg

    def beam_search_eval(self, epoch=None):
        best_acc = -np.inf
        # best_loss = np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_arch = None
        best_logits = None
        cand_seq, logits_seq, logP_accum, entropy_accum = self.model.arch_master.beam_search(self.args.top_seq)
        logging.info("beam search done. size: %d" % int(self.args.top_seq))
        for idx, cand in enumerate(cand_seq):
            # enumerate over each cand arch and perform testing
            top1 = AverageMeter('cand top1', ':3.3f')
            # cand = torch.tensor([v.item() for v in cand]).to(self.device)
            arch = [self.model.candidate_width[v] for v in cand]
            for test_input, test_target in self.test_loader:
                test_input, test_target = test_input.to(self.device), test_target.to(self.device)
                logits, _ = self.model.test_forward(test_input, cand)
                acc = accuracy(logits, test_target)[0]
                top1.update(acc.item(), test_input.size(0))

            flops = self.model.arch_master._compute_flops(cand)
            if flops > self.args.max_flops * 1.1 and self.args.flops:
                logging.info("Flops: %e larger than the threshold: %e, skipped" % \
                    (flops, 1.1*self.args.max_flops))
            else:
                logging.info("Evaluating the %d-th arch: %s" % (idx, str(arch)))
                logging.info("Arch id:%d, %s, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                    (idx, str(cand.tolist()), top1.avg, logP_accum[idx].item(), entropy_accum[idx].item(), flops))    # print all the sampled archs

            if top1.avg > best_acc:
                best_acc = top1.avg
                best_arch = arch
                best_arch_logP = logP_accum[idx].item()
                best_arch_ent = entropy_accum[idx].item()

        avg_logits = [torch.stack(logits) for logits in logits_seq]
        avg_logits = torch.stack(avg_logits).mean(0)
        avg_arch_info, avg_discrepancy = self.model.get_arch_info(avg_logits)
        logging.info(avg_arch_info)
        logging.info("Best: Accuracy %f -LogP %f ENT %f",best_acc,
                         -best_arch_logP, best_arch_ent)
        logging.info("Best Arch: %s" % str(best_arch))

    def epoch_eval(self, epoch):
        best_acc = -np.inf
        # best_loss = np.inf
        best_arch_logP = None
        best_arch_ent = None
        best_arch = None
        best_logits = None
        avg_logits_list = []
        # sample 20 archs and take the best one.
        for i in range(self.args.n_test_archs):
            top1, aux_top1s, arch_cand, arch_logP, arch_entropy, arch_info, discrepancy = self.model.test_cand_arch(self.test_loader)
            flops = self.model.arch_master._compute_flops(arch_cand)
            logging.info("Arch id:%d, %s, Acc:%.3f, log P:%.4e, entropy:%.4e, flops:%e" % \
                (i, str(arch_cand.tolist()), top1, arch_logP.item(), arch_entropy.item(), flops))    # print all the sampled archs
            avg_logits_list.append(self.model.logits)
            # NOTE: print aux accs
            if self.args.use_aux:
                for layer_id, top1s in enumerate(aux_top1s):
                    tmp_str = 'layer: %d \n' % layer_id
                    for cand_id, acc in enumerate(top1s):
                        tmp_str += 'arch: %d, acc: %.3f || ' \
                            % (self.model.candidate_width[cand_id], acc)
                    logging.info(tmp_str)

            if top1 > best_acc:
                best_acc = top1
                best_arch = arch_cand
                best_arch_logP = arch_logP
                best_arch_ent = arch_entropy
                best_logits = arch_info

        avg_logits = torch.stack(avg_logits_list)
        avg_arch_info, avg_discrepancy = self.model.get_arch_info(avg_logits.mean(0))
        logging.info(avg_arch_info)
        logging.info("Best: Accuracy %f -LogP %f ENT %f",best_acc,
                         -best_arch_logP, best_arch_ent)
        logging.info("Best Arch: %s", decode_cfg(self.args, best_arch))

    def __setup_optim(self):
        """ Set up optimizer for network parameters (not arch parameters) """
        if self.args.model_type.startswith('resnet_'):
            self.opt = optim.SGD([{'params':list(self.model.meta_parameters())},
                                  {'params':list(self.model.projection_parameters()), 'lr':self.proj_lr}], lr=self.args.lr, \
                                 momentum=self.args.momentum, nesterov=self.args.nesterov, \
                                 weight_decay=self.args.weight_decay)
            # print('The projection params are:', [k for k, v in self.model.named_meta_parameters()])
            if self.args.lr_decy_type == 'multi_step':
                self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[\
                    int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)])
            elif self.args.lr_decy_type == 'cosine':
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(\
                    self.opt, self.args.epochs, eta_min=self.args.lr_min)

            self.aux_opt = optim.SGD(self.aux_parameeters, lr=self.args.lr, \
                                 momentum=self.args.momentum, nesterov=self.args.nesterov, \
                                 weight_decay=self.args.arch_weight_decay)
            self.aux_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.aux_opt, milestones=[\
                    int(self.args.epochs * 0.5), int(self.args.epochs * 0.75)])

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



