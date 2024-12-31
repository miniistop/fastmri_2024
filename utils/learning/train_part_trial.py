import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
import pathlib
import optuna
import torch.optim as optim
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from pathlib import Path
import copy
from utils.own.time_util import hr_min_sec_print, finish_time
import fastmri
import cv2
from torch.nn import functional as F
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet
from utils.own.extension import print_memory_usage
from utils.model.fgdnet import FGDNet
from utils.model.CombinedModel import CombinedModel, VarNoiseNet, VarMWCNN, ADNet
from utils.model.denoiser.mwcnn import make_model
from utils.model.model_utils import load_best_model
from utils.model.eamri import EAMRI
import os
from utils.data.mask import get_mask, shift_mask
########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################    
def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, previous_model , model_num):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    train_loss_log_inepoch = np.empty((0,2))
    
#     for i in range(7):
    for iter, data in enumerate(data_loader):
#         input, target, maximum, fnames, slices= data
#         input = input.cuda(non_blocking=True)
        mask, full_kspace, kspace, target, maximum, fnames, _ = data

#             for _ in range(i):
#                 mask = shift_mask(mask)
#                 kspace = full_kspace*mask
        kspace = kspace.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        target = target.cuda(non_blocking = True)
        output,_,_,_= model(kspace, mask)
        loss = loss_type(output, target, maximum)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
            
        train_loss_log_inepoch = np.append(train_loss_log_inepoch, np.array([[int(iter), total_loss/(iter+1)]]), axis=0)
            
            
            
        if iter == 2:
            ex_time = time.perf_counter()-start_iter
            finish_time(args.num_epochs, len(data_loader), ex_time)

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} 'f'SSIM = {ssss} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
            
    total_loss = total_loss / (len_loader)
    
    file_path = os.path.join(args.val_loss_dir, f"train_loss_log_epoch{epoch+1}")
    np.save(file_path, train_loss_log_inepoch)
    print(f"loss file saved! {file_path}")
    
    return total_loss, time.perf_counter() - start_epoch
#######################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

def validate(args, model, data_loader, previous_model , model_num):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            
            mask, full_kspace, kspace, target, _, fnames, slices= data

            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            target_edge = get_edge(target).cuda(non_blocking = True)
            target = target.cuda(non_blocking = True)
            output,_,_,_= model(kspace, mask)
            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, model_num, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'model_num':model_num,
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / f'model{model_num}.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / f'model{model_num}.pt', exp_dir / f'best_model{model_num}.pt')

########################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################

def train(args, model_num):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    # model = EAMRI()
    model = VarNet(num_cascades=args.cascade, chans=args.chans, sens_chans=args.sens_chans)
    previous_model = -1
    model.to(device=device)

    train_loader = create_data_loaders(isimage = False, input_key = "kspace", data_path = args.data_path_train, args = args, shuffle=True, acc = 8, to_acc = 0)
    
    val_loader = create_data_loaders(isimage = False, input_key = "kspace", data_path = args.data_path_val, args = args, shuffle=True, acc=8, to_acc = 0)

    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    best_val_loss = 1.
    start_epoch = 0
    val_loss_log = np.empty((0, 2))
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, previous_model, model_num)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader, previous_model , model_num)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, f"val_loss_log_{model_num}")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)


        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, model_num , optimizer, best_val_loss, is_new_best)
        print(
            f'Model{1} : '
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
    
    return train_loss, best_val_loss


def objective(args, trial):
    # Define the range of hyperparameters to search over
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    momentum = trial.suggest_uniform("momentum", 0.0, 1.0) 
    cascade = trial.suggest_uniform("momentum", 1,11) 
    chans = trial.suggest_uniform("momentum", 9, 18) 
    sens_chans = trial.suggest_uniform("momentum", 4, 8) 

    
    # Define the loss function and optimizer
    model = VarNet(num_cascades=cascade, chans=chans, sens_chans=sens_chans)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, momentum=momentum)
    args.cascade = cascade
    args.chans = sens_chans
    args.sens_chans

    
    train(args, 1)
    
        
    return accuracy

# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the best hyperparameters and accuracy
print("Best hyperparameters: ", study.best_params)
print("Best accuracy: ", study.best_value)