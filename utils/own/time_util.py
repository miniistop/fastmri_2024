import torch
import argparse
import shutil
import os, sys
from pathlib import Path
import time

def hr_min_sec_make(time):
    hour = time // 3600
    min = ( time % 3600 ) // 60
    sec = ( time % 3600 ) % 60
    return hour, min, sec

def hr_min_sec_print(label, time):
    hour, min, sec = hr_min_sec_make(time)
    print(f"%%%%%%%%%%%%%%%%%%%%%%%%%% {label} completed %%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("--------------------------------------------------------")
    print(f"{label}_time is {int(hour)} hr : {int(min)} min : {int(sec)} sec")
    print("--------------------------------------------------------")

def finish_time(epoch, iter, time):
    time = time*(epoch*iter)-time
    hour, min, sec = hr_min_sec_make(time)
    print("--------------------------------------------------------")
    print(f"estimated_finish_time is {int(hour)} hr : {int(min)} min : {int(sec)} sec later")
    print("--------------------------------------------------------")