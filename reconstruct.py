import argparse
from pathlib import Path
import os, sys
import time
from utils.own.time_util import hr_min_sec_print

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.test_part import forward

    
# def parse():
#     parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
#                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
#     parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
#     parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
#     parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')
    
#     parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
#     parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
#     parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
#     parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

#     args = parser.parse_args()
#     return args


def reconstruct_py(args, model_num):

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    
    start_time = time.perf_counter() ##추가한 부분
    
    # acc4
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc4"
    print(args.forward_dir)
    forward(args, model_num)
    
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'reconstructions_leaderboard' / "acc8"
    print(args.forward_dir)
    forward(args, model_num)
    
    acc8_time = time.perf_counter()
    acc8_time = acc8_time - start_time


    hr_min_sec_print("total_recon", acc8_time)
