

import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms
import numpy as np

from unet import Unet
from fgdnet import FGDNet
from bio_unet import BiONet
from varnet import NormUnet, NormUnetAtt
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
            [CDFFNet2(chans = chans, num_pools = pools) for _ in range(num_cascades)]
        )

    def forward(self, result_img: torch.Tensor) -> torch.Tensor:
        origin_img = result_img.clone()
        origin_kspace = torch.fft.fftn(result_img)
        for cascade in self.cascades:
            result_img= cascade(result_img, origin_img, origin_kspace)

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
        self.dc_weight = nn.Parameter(torch.tensor(1.))
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
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:

        image1 = self.img_model(image)
        kspace1 = torch.fft.fftn(image1)
        kspace2 = (kspace1 - origin_kspace)
        kspace2 = torch.stack((kspace2.real, kspace2.imag), dim=-1).permute(0,3,1,2)
        kspace3 = self.kspace_model(kspace2).permute(0,2,3,1)
        kspace3 = (kspace3[..., 0] + 1j * kspace3[..., 1])*self.dc_weight + kspace1
        image4 = torch.fft.ifftn(kspace3)
        image4 = torch.abs(image4)
    
        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image4

class CDFFNet(nn.Module):
    
    def __init__(self, chans: int, num_pools: int):

        super().__init__()
        
        self.img_model = NormUnetAtt(chans = 9, num_pools = 6)
        self.kspace_model = MWCNN()
        
    def forward(
        self,
        image: torch.Tensor,
        origin_image: torch.Tensor,
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
              
        kspace_input = torch.fft.fftn(image)
        kspace_input = torch.stack((kspace_input.real, kspace_input.imag), dim = -1).permute(0, 3, 1, 2)
        kspace_inter = self.kspace_model(kspace_input).permute(0, 2, 3, 1)
        kspace_inter = kspace_inter[..., 0] + 1j * kspace_inter[..., 1]
        
        img_input = torch.fft.ifftn(kspace_inter)
        img_input = torch.stack((img_input.real, img_input.imag), dim = -1).unsqueeze(0)
        img_inter = self.img_model(img_input).squeeze(0)
        img_inter = img_inter[..., 0] + 1j * img_inter[..., 1]
        img_inter = torch.abs(img_inter)
        
        output = fusion_SFNN(abs(img_inter), abs(torch.fft.ifftn(kspace_inter)), abs(image))
        
        return output

    
class CDFFNet2(nn.Module):
# 1,000 iter: train_loss = 0.034
    
    def __init__(self, chans: int, num_pools: int):

        super().__init__()
        
        self.img_model = NormUnetAtt(chans = 9, num_pools = 6)
        self.kspace_model = MWCNN()
        self.dc_weight1 = nn.Parameter(torch.tensor(0.6))
        self.dc_weight2 = nn.Parameter(torch.tensor(0.6))
        
        
    def forward(
        self,
        image: torch.Tensor,
        origin_image: torch.Tensor,
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = image
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight1
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        image3 = torch.abs(image2)
        
        image2 = torch.stack((image2.real, image2.imag), dim = -1).unsqueeze(0)
        image4 = self.img_model(image2).squeeze(0)
        image4 = image4[..., 0] + 1j * image4[..., 1]
        image4 = torch.abs(image4)
        image_output = image3 + image4
        
        image4 = (image_output - origin_image)*self.dc_weight2
#         print(image4.dtype)
#         print(origin_image.dtype)
        image_output = image_output - image4
        image_fusion = fusion_SFNN(abs(image_output), abs(torch.fft.ifftn(kspace_output)), abs(image))
#         print(image_output.dtype)
        print(self.dc_weight1, self.dc_weight2)
        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image_fusion
    
    
    
class XPDNetBlock_Unet_Fusion(nn.Module):
# 1,000 iter: train_loss = 0.034
    
    def __init__(self, chans: int, num_pools: int):

        super().__init__()
        
        self.img_model = NormUnetAtt(chans = 9, num_pools = 6)
        self.kspace_model = MWCNN()
        self.dc_weight1 = nn.Parameter(torch.tensor(0.6))
        self.dc_weight2 = nn.Parameter(torch.tensor(0.6))
    def forward(
        self,
        image: torch.Tensor,
        origin_image: torch.Tensor,
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = image
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight1
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        image3 = torch.abs(image2)
        
        image2 = torch.stack((image2.real, image2.imag), dim = -1).unsqueeze(0)
        image4 = self.img_model(image2).squeeze(0)
        image4 = image4[..., 0] + 1j * image4[..., 1]
        image4 = torch.abs(image4)
        image_output = image3 + image4
        
        image4 = (image_output - origin_image)*self.dc_weight2
#         print(image4.dtype)
#         print(origin_image.dtype)
        image_output = image_output - image4
        image_fusion = fusion_strategy(image_output, origin_image)
#         print(image_output.dtype)
        print(self.dc_weight1, self.dc_weight2)
        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image_fusion
    
    
class XPDNetBlock_Unet(nn.Module):

    def __init__(self, chans: int, num_pools: int):

        super().__init__()
        
        self.img_model = NormUnetAtt(chans = 9, num_pools = 6)
        self.kspace_model = MWCNN()
        self.dc_weight1 = nn.Parameter(torch.tensor(1.))
        self.dc_weight2 = nn.Parameter(torch.tensor(1.))
        
    def forward(
        self,
        image: torch.Tensor,
        origin_image: torch.Tensor,
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = image
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight1
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        image3 = torch.abs(image2)
        
        image2 = torch.stack((image2.real, image2.imag), dim = -1).unsqueeze(0)
        image4 = self.img_model(image2).squeeze(0)
        image4 = image4[..., 0] + 1j * image4[..., 1]
        image4 = torch.abs(image4)
        image_output = image3 + image4
        
        image4 = (image_output - origin_image)*self.dc_weight2
#         print(image4.dtype)
#         print(origin_image.dtype)
        image_output = image_output - image4
#         print(image_output.dtype)
        print(self.dc_weight1, self.dc_weight2)
        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image_output    
        
    
class XPDNetBlock3(nn.Module):

    def __init__(self, chans: int, num_pools: int):

        super().__init__()
        
        self.img_model = FGDNet()
        self.kspace_model = MWCNN()
        self.dc_weight1 = nn.Parameter(torch.tensor(1.))
        self.dc_weight2 = nn.Parameter(torch.tensor(1.))
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
        origin_image: torch.Tensor,
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = image
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight1
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        image = torch.abs(image2)
        
        image2 = torch.stack((image2, torch.zero_like(image2)), dim = -1)
        image3 = self.img_model(image2)
        image_output = image2 + image3
        
        image4 = (image_output - origin_image)*self.dc_weight2
        image_output = image_output - image4

        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image_output

class XPDNetBlock2(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.img_model = FGDNet()
        self.kspace_model = MWCNN()
        self.dc_weight = nn.Parameter(torch.tensor(1.))
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
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = self.img_model(image)
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        image2 = torch.abs(image2)

        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        return image2    
    
class XPDNetBlock4(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.img_model = FGDNet()
        self.kspace_model = MWCNN()
        self.dc_weight = nn.Parameter(torch.tensor(1.))
#         self.boundary = 132* nn.Parameter(torch.tensor(1.0))
        self.conv = nn.Conv2d(2, 1, kernel_size = 3, stride = 2, padding = 1)

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
        origin_kspace: torch.Tensor
    ) -> torch.Tensor:
        
        image1 = self.img_model(image)
        kspace2 = torch.fft.fftn(image1)
        kspace3 = torch.stack((kspace2.real, kspace2.imag), dim = -1).permute(0, 3, 1, 2)
        kspace4 = self.kspace_model(kspace3).permute(0, 2, 3, 1)
        kspace4 = kspace4[..., 0] + 1j * kspace4[..., 1]
        kspace_output = kspace2 + kspace4
        
        print(kspace_output.shape)
        print(origin_kspace.shape)

        kspace5 = (kspace_output - origin_kspace)*self.dc_weight
        kspace_output = kspace_output - kspace5

        image2 = torch.fft.ifftn(kspace_output)
        image3 = torch.cat([image1, image2], dim = 1) 
        print(image3.shape)
        
        image3 = torch.stack((image3.real, image3.imag), dim = -1).permute(0, 3, 1, 2)
        print(image3.shape)
        
        image3 = self.conv(image3)
        print(image3.shape)
        image3 = torch.squeeze(image3, dim = -1)
        # image_real = self.conv(image3.real.unsqueeze(2))
        # image_imag = self.conv(image3.imag.unsqueeze(2))
        
        # image_real = image_real.squeeze(2)
        # image_imag = image_imag.squeeze(2)
        
        # print(image_real.shape)
        
        # image3 = torch.cat([image_real, image_imag], dim = 1)
        image3 = torch.abs(image3)
        print(image3.shape)
        
        return image3
                
"""
        resize_dims1 = (kspace_output.shape[-2], kspace_output.shape[-1])
        origin_kspace_resized = torch.nn.functional.interpolate(origin_kspace.unsqueeze(0), size=resize_dims1, mode='bilinear', align_corners=False)
        origin_kspace_resized = origin_kspace_resized.squeeze(0)
        
        kspace5 = (kspace_output - origin_kspace)*self.dc_weight
        kspace_output = kspace_output - kspace5
        
        image2 = torch.fft.ifftn(kspace_output)
    
        # Resize image2 to match the size of image1
        resize_dims2 = (image1.shape[-2], image1.shape[-1])
        image2_real = torch.stack([image2.real, image2.imag], dim=-1)
        image2_real_resized = torch.nn.functional.interpolate(image2_real.permute(0, 2, 3, 1), size=resize_dims2, mode='bilinear', align_corners=False)
        image2_real_resized = image2_real_resized.permute(0, 2, 3, 1)

        image2_resized = image2_real_resized[..., 0] + 1j * image2_real_resized[..., 1]
        image3 = torch.cat([image1, image2_resized], dim=1)
        image3 = torch.abs(image3)
        
        


        # image2 = torch.fft.ifftn(kspace_output)
        # image3 = torch.cat([image1, image2], dim = 1)
        # image3 = torch.abs(image3)

        # out_image = image4*self.dc_weight + image*(1-self.dc_weight)
        # print(self.dc_weight)
        # return image3       

"""

def process_for_nuc(f):
            f = f.squeeze(0)
            total = []
            for i in range(f.shape[0]):
                temp = torch.norm(f[i])
                # total = np.append(total, temp)
                total.append(temp.item())
            return total   
        
        
def fusion_strategy(f1, f2, strategy="SFNN"):
    """
    f1: the extracted features of images 1
    f2: the extracted features of images 2
    strategy: 6 fusion strategy, including:
    "addition", "average", "FER", "L1NW", "AL1NW", "FL1N"
    addition strategy
    average strategy
    FER strategy: Feature Energy Ratio strategy
    L1NW strategy: L1-Norm Weight Strategy
    AL1NW strategy: Average L1-Norm Weight Strategy
    FL1N strategy: Feature L1-Norm Strategy

    Note:
    If the original image is PET or SPECT modal,
    it should be converted into YCbCr data, including Y1, Cb and Cr.
    """

    # The fused feature
    fused = torch.zeros_like(f1)
    if strategy == "addition":
        fused = f1 + f2
    elif strategy == "average":
        fused = (f1 + f2) / 2
    elif strategy == "FER":
        f_sum = (f1 ** 2 + f2 ** 2).clone()
        f_sum[f_sum == 0] = 1
        k1 = f1 ** 2 / f_sum
        k2 = f2 ** 2 / f_sum
        fused = k1 * f1 + k2 * f2
    elif strategy == "L1NW":
        l1 = l1_norm(f1)
        print(l1)
        l2 = l1_norm(f2)
        print(l2)
        fused = l1 * f1 + l2 * f2
    elif strategy == "AL1NW":
        p1 = l1_norm(f1) / 2
        p2 = l1_norm(f2) / 2
        fused = p1 * f1 + p2 * f2
    elif strategy == "FL1N":
        l1 = l1_norm(f1)
        l2 = l1_norm(f2)
        w1 = l1 / (l1 + l2)
        w2 = l2 / (l1 + l2)
        fused = w1 * f1 + w2 * f2
    elif strategy == "SFNN":
        f1_soft = nn.functional.softmax(f1)
        f2_soft = nn.functional.softmax(f2)
        l1 = process_for_nuc(f1_soft)
        #print(l1)
        l2 = process_for_nuc(f2_soft)
        l1 = np.array(l1)
        l2 = np.array(l2)
        # w1 = np.mean(l1) / (np.mean(l1) + np.mean(l2))
        # w2 = np.mean(l2) / (np.mean(l1) + np.mean(l2))
        # w1 = sum(l1) / (sum(l1) + sum(l2))
        # w2 = sum(l2) / (sum(l1) + sum(l2))
        w1 = max(l1)**2 / (max(l1)**2 + max(l2)**2)
        w2 = max(l2)**2 / (max(l1)**2 + max(l2)**2)
        # f_sum = (f1 ** 2 + f2 ** 2).clone()
        # f_sum[f_sum == 0] = 1
        # k1 = f1 ** 2 / f_sum
        # k2 = f2 ** 2 / f_sum

        fused = w1 * f1 + w2 * f2
        
    # Need to do reconstruction on "fused"
    return fused


def fusion_SFNN(f1, f2, f3):
    f1_soft = nn.functional.softmax(f1)
    f2_soft = nn.functional.softmax(f2)
    f3_soft = nn.functional.softmax(f3)
    
    l1 = process_for_nuc(f1_soft)
    l2 = process_for_nuc(f2_soft)
    l3 = process_for_nuc(f3_soft)
    l1 = max(np.array(l1)) ** 2
    l2 = max(np.array(l2)) ** 2
    l3 = max(np.array(l3)) ** 2
    
    w1 = l1 / (l1 + l2 + l3)
    w2 = l2 / (l1 + l2 + l3)
    w3 = l3 / (l1 + l2 + l3)
    
    fused = w1 * f1 + w2 * f2 + w3 * f3
    
    return fused
    
