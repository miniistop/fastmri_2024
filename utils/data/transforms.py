import numpy as np
import torch
from utils.data.mask import get_mask
import random
import math
from utils.data.mask import get_mask, shift_mask

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, acc=0, target_acc=0, train = True):
        self.isforward = isforward
        self.max_key = max_key
        self.acc = acc
        self.target_acc = target_acc
        self.train = train
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1 
            maximum = -1
#         mask = get_mask(len(mask), 8)
#         if (self.acc == 4)&("acc8" in fname)
#             mask = get_mask(len(mask), 4)
#         if self.target_acc == 0:
#             target_kspace = input
#             target_kspace = torch.stack((target_kspace.real, target_kspace.imag), dim=-1)
#         else:
#             target_kspace = to_tensor(input*get_mask(len(mask), self.target_acc ))
#             target_kspace = torch.stack((target_kspace.real, target_kspace.imag), dim=-1)    
#         if (self.targret_acc == 0)&(self.acc == 8)
#             acc4_kspace = to_tensor(input* get_mask(len(mask), 4))
#             acc4_kspace = torch.stack((acc4_kspace.real, acc4_kspace.imag), dim=-1)
#         else:
#             acc4_kspace = -1
        if self.train:
#             mask_change = random.random()
            
#             if "acc4" in fname:
#                 if mask_change>0.4:
#                     mask = get_mask(len(mask), 8)
#             elif "acc8" in fname:
                
#                 if mask_change>0.6:
#                     mask = get_mask(len(mask), 4)
            mask_aug = random.random()
            
            if mask_aug < 0.7:
                for i in range(int(math.ceil(mask_aug*10))):
                    mask = shift_mask(mask)
        target_kspace = to_tensor(input)
        target_kspace = torch.stack((target_kspace.real, target_kspace.imag), dim=-1)
        kspace = to_tensor(input*mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        return mask, target_kspace , kspace, target, maximum, fname, slice
 
    
####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################  

class DataTransform_img:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input, target, maximum, fname, slice
    
    