

import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from unet import Unet
from fgdnet import FGDNet
from bio_unet import BiONet
from varnet import NormUnet
from denoiser.mwcnn import MWCNN


class MyNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()
        

        self.cascades = nn.ModuleList(
            [myNetBlock(NormUnet(chans = 9, num_pools = 8)) for _ in range(num_cascades)]
        )

    def forward(self, result_img: torch.Tensor) -> torch.Tensor:
        origin_kspace = torch.fft.fftn(result_img)
        for cascade in self.cascades:
            result_img, before_kspace = cascade(result_img)

        height = result_img.shape[-2]
        width = result_img.shape[-1]
        result_img = result_img[..., (height - 384) // 2 : 384 + (height - 384) // 2, (width - 384) // 2 : 384 + (width - 384) // 2]
        return result_img

    


class myNetBlock(nn.Module):

    def __init__(self, model: nn.Module):

        super().__init__()
        
        self.model1 = model
        self.model2 = model
        self.model3 = model
        self.dc_weight = nn.Parameter(torch.tensor(0.1))
#         self.boundary = 132* nn.Parameter(torch.tensor(1.0))

         

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )

    def forward(
        self,
        pred_kspace: torch.Tensor, 
        origin_kspace: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        
        
        boundary = 16
        boundary2 = 32
        
        width = pred_kspace.shape[-2]
        low_mask = torch.zeros(width).cuda(non_blocking=True)
        low_mask[(width - boundary) // 2 : boundary + (width - boundary) // 2] = 1
        mid_mask = torch.zeros(width).cuda(non_blocking=True)
        mid_mask[(width - boundary2) // 2 : boundary2 + (width - boundary2) // 2] = 1
        mid_mask = mid_mask - low_mask
        high_mask = 1 - mid_mask
        high_mask = high_mask.cuda(non_blocking=True)
        kspace_pred_high = pred_kspace*high_mask.unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        kspace_pred_mid = pred_kspace*mid_mask.unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        kspace_pred_low = pred_kspace*low_mask.unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        out_kspace = self.sens_expand(
            (self.model1(self.sens_reduce(kspace_pred_high, sens_maps)) + self.model3(self.sens_reduce(kspace_pred_mid, sens_maps)) + self.model2(self.sens_reduce(kspace_pred_low , sens_maps))), sens_maps
        )
        out_kspace = (out_kspace*self.dc_weight + origin_kspace*(1-self.dc_weight))
        return out_kspace




class XPDNetBlock(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.img_model = FGDNet()
        self.kspace_model = MWCNN()
        self.weight1 = nn.Parameter(torch.tensor(1.))
        self.weight2 = nn.Parameter(torch.tensor(1.))
#         self.boundary = 132* nn.Parameter(torch.tensor(1.0))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
    def forward(
        self,
        image: torch.Tensor,
        before_kspace: torch.Tensor = None
    ) -> torch.Tensor:
        
        kspace2 = torch.fft.fftn(image)
        if not before_kspace == None:
            kspace2 = kspace2*self.weight1 + before_kspace*(1-self.weight1)
        kspace2 = torch.stack((kspace2.real, kspace2.imag), dim=-1).permute(0,3,1,2)
        kspace3 = self.kspace_model(kspace2).permute(0,2,3,1)
        kspace3 =(kspace3[..., 0] + 1j * kspace3[..., 1]
        image4 = torch.fft.ifftn(kspace3)
        image4 = image4*self.weight2 + image*(1-self.weight2)
        image5 = self.img_model(image4)
        
        return image5, kspace3

    
