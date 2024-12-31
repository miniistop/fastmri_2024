import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from unet import Unet
from fgdnet import FGDNet, Noise_Est
from varnet import VarNet
from denoiser.mwcnn import MWCNN


class CombinedModel(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        super(CombinedModel, self).__init__()
        
        self.num_cascades = num_cascades
        self.chans = chans
        self.sens_chans = sens_chans
        
        self.model1 = VarNet(num_cascades, chans, sens_chans)
        self.model2 = FGDNet()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        x = self.model1(masked_kspace, mask)
        x_squeezed = x.unsqueeze(dim=0)
        guidance_squeezed = guidance.unsqueeze(dim=0)
        x = self.model2(x_squeezed, guidance_squeezed)
        x = x.squeeze(dim=0)
        return x
    
class VarNoiseNet(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        super(VarNoiseNet, self).__init__()
        
        self.num_cascades = num_cascades
        self.chans = chans
        self.sens_chans = sens_chans
        
        self.model1 = VarNet(num_cascades, chans, sens_chans)
        self.model2 = Noise_Est()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.model1(masked_kspace, mask)
        x_squeezed = x.unsqueeze(dim=0)
        x = self.model2(x_squeezed)
        x = x.squeeze(dim=0)
        return x
    
    
class VarMWCNN(nn.Module):
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
    ):
        super(VarMWCNN, self).__init__()
        
        self.num_cascades = num_cascades
        self.chans = chans
        self.sens_chans = sens_chans
        
        self.model1 = VarNet(num_cascades, chans, sens_chans)
        self.model2 = MWCNN()

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.model1(masked_kspace, mask)
        x_squeezed = x.unsqueeze(dim=0)
        x = self.model2(x_squeezed)
        x = x.squeeze(dim=0)
        return x
    
    
class ADNet(nn.Module):
    def __init__(
        self
    ):
        super(ADNet, self).__init__()
        
        
        self.model = Noise_Est()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_squeezed = x.unsqueeze(dim=0)
        x = self.model(x_squeezed)
        x = x.squeeze(dim=0)
        return x