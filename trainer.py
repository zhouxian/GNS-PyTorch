import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import tprint
from timeit import default_timer as timer
from config import _C as C
from torch.utils.tensorboard import SummaryWriter
import utils

class Trainer(object):
    def __init__(self, device, train_loader, val_loader, model, optim,
                 max_iters, exp_name):
        self.device = device
        self.exp_name = exp_name

        self.train_loader, self.val_loader = train_loader, val_loader
        self.metadata = utils.update_metadata(self.train_loader.dataset.metadata, self.device)

        self.model = model
        self.optim = optim
        self.iterations = 0
        self.max_iters = max_iters
        self.val_interval = C.SOLVER.VAL_INTERVAL

        self._setup_loss()
        self.setup_dirs()

        self.tb_writer = SummaryWriter(self.log_dir)

    def setup_dirs(self):
        self.log_dir = f'./logs/{self.exp_name}'
        self.ckpt_dir = f'./ckpts/{self.exp_name}'

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)


    def train(self):
        print_msg = "| ".join(["progress "] + list(map("{:6}".format, self.loss_name)))
        self.model.train()
        print('\r', end='')
        print(print_msg)
        self.best_val_pos_loss = 1e7
        while self.iterations < self.max_iters:
            self.train_epoch()

    def train_epoch(self):
        self.tt = timer()
        for batch_idx, data in enumerate(self.train_loader):
            # only support B=1 for now
            assert data[0].shape[0] == 1

            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss = data

            self._adjust_learning_rate()
            self.optim.zero_grad()

            num_rollouts = tgt_vels.shape[1]
            outputs = self.model(poss, particle_type, self.metadata, nonk_mask, tgt_poss, num_rollouts=num_rollouts, phase='train')
            
            tgt_accns = (tgt_accs - self.metadata['acc_mean'])/self.metadata['acc_std']

            labels = {
                'accns': tgt_accns,
                'poss': tgt_poss
            }
            loss = self.loss(outputs, labels, nonk_mask, 'train')
            loss.backward()
            self.optim.step()

            self.iterations += 1

            print_msg = ""
            print_msg += f"{self.iterations}"
            print_msg += f" | "
            print_msg += f" | ".join(
                ["{:.2f}".format(self.single_losses[name]) for name in self.loss_name])
            speed = self.loss_cnt / (timer() - self.time)
            step_time = timer() - self.tt
            self.tt = timer()
            eta = (self.max_iters - self.iterations) / speed / 3600
            print_msg += f" | speed: {speed:.2f} | step_time: {step_time:.2f} | eta: {eta:.2f} h"
            tprint(print_msg)

            # logging to tb
            for name in self.loss_name:
                self.tb_writer.add_scalar(f'Single/{name}', self.single_losses[name], self.iterations)
                self.tb_writer.add_scalar(f'Period/{name}', self.period_losses[name] / self.loss_cnt, self.iterations)

            if self.iterations % self.val_interval == 0:
                self.snapshot(self.iterations)
                self.val()
                self._init_loss()
                self.model.train()

            if self.iterations >= self.max_iters:
                break

    def val(self):
        self.model.eval()
        self._init_loss()
        print()

        for batch_idx, data in enumerate(self.val_loader):
            # only support B=1 for now
            assert data[0].shape[0] == 1

            for i in range(len(data)):
                data[i] = data[i][0].to(self.device)
            poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss = data

            tprint(f'eval: {batch_idx}/{len(self.val_loader)}')
            
            with torch.no_grad():

                num_rollouts = tgt_vels.shape[1]
                outputs = self.model(poss, particle_type, self.metadata, nonk_mask, tgt_poss, num_rollouts=num_rollouts, phase='test')

                tgt_accns = (tgt_accs - self.metadata['acc_mean'])/self.metadata['acc_std']
                    
                labels = {
                    'accns': tgt_accns,
                    'poss': tgt_poss
                }
                self.loss(outputs, labels, nonk_mask, 'test')

                if outputs['pred_collaposed']:
                    break

        # logging to tb
        for name in self.loss_name:
            self.tb_writer.add_scalar(f'Val/{name}', self.period_losses[name] / self.loss_cnt, self.iterations)

        # save best model
        if self.period_losses['pos'] < self.best_val_pos_loss:
            self.best_val_pos_loss = self.period_losses['pos']
            self.snapshot_best(self.iterations)

    def loss(self, outputs, labels, weighting, phase):
        self.loss_cnt += 1

        if outputs['pred_collaposed']:
            for name in self.loss_name:
                self.single_losses[name] = np.nan
                self.period_losses[name] = np.nan
            loss = np.nan
            return loss

        accn_loss = ((outputs['pred_accns'] - labels['accns']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        accn_loss = accn_loss.mean(1).sum() / torch.sum(weighting)
        self.single_losses['accn'] = accn_loss.item()
        self.period_losses['accn'] += self.single_losses['accn']

        pos_loss = ((outputs['pred_poss'] - labels['poss']) * torch.unsqueeze(torch.unsqueeze(weighting, -1), 1)) ** 2
        pos_loss = pos_loss.mean(1).sum() / torch.sum(weighting)
        self.single_losses['pos'] = pos_loss.item()
        self.period_losses['pos'] += self.single_losses['pos']

        loss = accn_loss

        return loss

    def snapshot(self, iter):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.ckpt_dir, f'iter_{iter}.path.tar'),
        )

    def snapshot_best(self, iter):
        torch.save(
            {
                'arch': self.model.__class__.__name__,
                'model': self.model.state_dict(),
            },
            os.path.join(self.ckpt_dir, f'best_{iter}.path.tar'),
        )

    def _setup_loss(self):
        # normalized acceleration loss and position loss
        self.loss_name = ['accn', 'pos']
        self._init_loss()

    def _init_loss(self):
        self.single_losses = dict.fromkeys(self.loss_name, 0.0)
        self.period_losses = dict.fromkeys(self.loss_name, 0.0)
        self.loss_cnt = 0
        self.time = timer()

    def _adjust_learning_rate(self):
        if self.iterations < C.SOLVER.WARMUP_ITERS:
            lr = C.SOLVER.BASE_LR * self.iterations / C.SOLVER.WARMUP_ITERS
            
        else:
            lr = (C.SOLVER.BASE_LR - C.SOLVER.MIN_LR) * 0.1 ** (self.iterations / C.SOLVER.LR_DECAY_INTERVAL) + C.SOLVER.MIN_LR

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr
