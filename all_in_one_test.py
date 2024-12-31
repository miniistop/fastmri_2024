import torch
import argparse
import shutil
import os, sys
import time
from pathlib import Path
import pathlib
temp = pathlib.WindowsPath
pathlib.WindowsPath = pathlib.PosixPath
from utils.own.time_util import hr_min_sec_print
from train import train_py
from reconstruct import reconstruct_py
from leaderboard_eval import leaderboard_eval_py
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=3.3e-4, help='Learning rate') #0.00033
    parser.add_argument('-r', '--report-interval', type=int, default=1, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    parser.add_argument('--optim_batch', type=int, default=1, help='Batch size')

    
    ####하이퍼 파라미터 조정값####
    
    parser.add_argument('--end', type=int, default = 1)
    parser.add_argument('--interval', type=int,default = 1)
    
    parser.add_argument('--cascade', type=int, default=11, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    
    # 11 10 6
    
    
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=2023, help='Fix random seed')
    
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
    parser.add_argument('-lp', '--path_leaderboard_data', type=Path, default='/Data/leaderboard/')
    
    """
    Modify Path Below To Test Your Results
    """
    parser.add_argument('-yp', '--path_your_data', type=Path, default='../result/test_varnet/reconstructions_leaderboard/')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')
    
    parser.add_argument('--edge_weight', type=float, default =1)
    parser.add_argument('--momentum' ,type = float, default = 6e-06)
    parser.add_argument('--load', type=bool , default = 0)
    parser.add_argument('--cascade2', type=int, default=1)
    parser.add_argument('--aug', type=bool , default = 0)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse()
    ssim_score = []
    print(f"num of channels : {args.chans}")
    print(f"num of sens channels : {args.sens_chans}")
#     train_py(args,2)
    train_py(args,2)
    reconstruct_py(args,2)
    ssim_score.append(leaderboard_eval_py(args))
    print(ssim_score)
"""
#     for i in range(9, 10, 4):
#         for j in range(6, 4, 4):
#             print(f"num of channels : {i}")
#             print(f"num of sens channels : {j}")
#             args.chans = i
#             args.sens_chans = j
#             train_py(args)
#             reconstruct_py(args)
#             ssim_score += leaderboard_eval_py(args)
#         print(ssim_score)
#     print(ssim_score)
"""
    







    
    
