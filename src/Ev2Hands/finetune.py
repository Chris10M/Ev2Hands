
import sys; sys.path.append('..') # add settings.
import os
import torch
import torch.nn as nn
import os
import collections
import sys
import time
import datetime

import arg_parser
from torch import optim
from torch.utils.data import DataLoader
from utils import Logger, set_model_logger, load_network
from model import TEHNetWrapper
from losses import Loss 
from evaluate import evaluate_net

from dataset import Ev2HandRDataset

from settings import REAL_TRAIN_DATA_PATH, REAL_TEST_DATA_PATH


def train(net, device):
    learning_rate = 0.001
    weight_decay = 0.0

    saved_path = os.getenv('CHECKPOINT_PATH', '')
    batch_size = int(os.getenv('BATCH_SIZE', 8))

    start_it = 0
    msg_iter = 50
    max_iter = 15000
    save_iter = 5000
    max_eval_iters = 8192

    set_model_logger(net.net)
    net.train()

    # setup data loader
    train_dataset = Ev2HandRDataset(REAL_TRAIN_DATA_PATH)
    validation_dataset = Ev2HandRDataset(REAL_TEST_DATA_PATH, augment=False)
    
    Logger.logger.info(f'''Cpu count: {os.cpu_count()}\nBatch size: {batch_size}\n''')

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=min(os.cpu_count(), batch_size), 
                              pin_memory=True)

    validation_loader = DataLoader(validation_dataset, 
                              batch_size=32, 
                              shuffle=True, 
                              num_workers=min(os.cpu_count(), batch_size), 
                              pin_memory=True)
                              
    optimizer = optim.Adam([{'params': net.parameters()}, 
                            ], lr=learning_rate, weight_decay=weight_decay)
    criterion = Loss(hands=net.hands, device=device).to(device)


    net, optimizer, start_it, max_eval_score = load_network(saved_path, net, optimizer, device, Logger.logger.info)

    epoch = 0
    diter = iter(train_loader)
    st = glob_st = time.time()
    loss_log = collections.defaultdict(float)
    save_step = False

    net.net = nn.DataParallel(net.net)

    for it in range(start_it, max_iter):
        try:
            batch = next(diter)
        except StopIteration:
            epoch += 1
            diter = iter(train_loader)
            batch = next(diter)
            save_step = True

        batch = Ev2HandRDataset.to_device(batch, device)
        
        lnes = batch['events']

        outputs = net(lnes)
        
        losses = criterion(outputs, batch)
        loss = sum(losses.values())

        losses['loss'] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in losses.items(): 
            if isinstance(v, torch.Tensor):
                loss_log[k] += v.item()
            else:
                loss_log[k] += v 

        if save_step or ((it + 1) % save_iter == 0):
            save_step = False
            
            net.net = net.net.module
            metrics, eval_score = evaluate_net(net, validation_loader, device, max_eval_iters)
            net.net = nn.DataParallel(net.net)

            Logger.logger.info(f"Model@{it + 1}\n Evaluation score: {eval_score}, Metrics: {metrics}")

            for k, v in metrics['auc'].items():
                Logger.tensorboard.add_scalar(k, v, epoch)

            now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            save_pth = os.path.join(Logger.ModelSavePath, f'{it + 1}_{eval_score}_{now}.pth')

            if eval_score > max_eval_score: 
                print(f'Saving model at: {(it + 1)}, save_pth: {save_pth}')
                torch.save({
                    'start_it': it,
                    'state_dict': net.state_dict(),
                    'optimize_state': optimizer.state_dict(),
                    'max_eval_score': max_eval_score,
                    'metrics': metrics,
                }, save_pth)
                print(f'model at: {(it + 1)} Saved')

                max_eval_score = eval_score

        #   print training log message
        if (it+1) % msg_iter == 0:
            for k, v in loss_log.items(): loss_log[k] = round(v / msg_iter, 2)

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    f'epoch: {epoch}',
                    'it: {it}/{max_it}',         
                    *[f"{k}: {v}" for k, v in loss_log.items()],
                    'eta: {eta}',
                    'time: {time:.2f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    time = t_intv,
                    eta = eta
                )
            Logger.logger.info(msg)
            
            loss_log = collections.defaultdict(float)
            st = ed


def main():
    arg_parser.finetune()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = TEHNetWrapper(device=device)
    train(net=net, device=device)


if __name__ == '__main__':
    main()
