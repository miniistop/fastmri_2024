"""
official network architecture for EAMRI
author: huihui
"""

import sys
sys.path.insert(0,'..')
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
import fastmri
from fastmri.data import transforms_simple as T
from torch.nn import functional as F
import pdb
import numpy as np
import math
from einops import rearrange
from torch.nn.parameter import Parameter
import numbers
from typing import List, Optional, Tuple



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x).contiguous()), h, w).contiguous()
    

class DC_multicoil(nn.Module):
    """
    data consistency layer for multicoil
    """
    def __init__(self, initLamda = 1, isStatic = True, shift=False):
        super(DC_multicoil, self).__init__()
        self.normalized = True #norm == 'ortho'
        self.lamda = Parameter(torch.Tensor(1))
        self.lamda.data.uniform_(0, 1)
        self.isStatic = isStatic
        self.shift = shift 
    
    def forward(self, xin, y, mask, sens_map):
        """
        xin: (B, 2, H, W)
        """
        mask_1 = 1
        assert xin.shape[1] == 2, "dc layer the last dimension of input x should be greater than 2"
        iScale = 1
        
        xin = T.expand_operator(xin.permute(0,2,3,1), sens_map, dim=1) #(B, coils, H, W, 2)
        xin_f = T.fft2(xin,normalized=self.normalized, shift=self.shift)
        xGT_f = y
        xout_f = xin_f + (- xin_f + xGT_f) * iScale * mask_1
        xout = T.ifft2(xout_f, normalized=self.normalized, shift=self.shift) 
        xout = T.reduce_operator(xout, sens_map, dim=1) #(B, H, W, 2)
        
        return xout.permute(0,3,1,2).contiguous()



def default_conv(in_channels, out_channels, kernel_size, dilate=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=dilate, dilation=dilate, bias=bias) 

class dilatedConvBlock(nn.Module):
    def __init__(self, iConvNum = 3, inChannel=32):
        super(dilatedConvBlock, self).__init__()
        self.LRelu = nn.LeakyReLU()
        convList = []
        for i in range(1, iConvNum+1):
            tmpConv = nn.Conv2d(inChannel,inChannel,3,padding = i, dilation = i)
            convList.append(tmpConv)
        self.layerList = nn.ModuleList(convList)
    
    def forward(self, x1):
        x2 = x1
        for conv in self.layerList:
            x2 = conv(x2)
            x2 = self.LRelu(x2)
        
        return x2


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
    


class rdn_convBlock(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 2, midChannel=16, shift=False):
        super(rdn_convBlock, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,midChannel,3,padding = 1)
        self.dilateBlock = dilatedConvBlock(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel,inChannel,3,padding = 1)
        self.dc = DC_multicoil(shift=shift) 


    def forward(self, x1, y, m, sens_map):
        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1 #(B, 2, H, W)
      
        # dc
        x5 = self.dc(x5, y, m, sens_map) #(B, coils, H, W, 2)

        return x5




class MSRB(nn.Module):
    def __init__(self, n_feats):
        """
        n_feats: input dimension
        """
        super(MSRB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        
        # stage 1
        self.conv_3_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, bias=True, dilation=2)
        self.fuse1 = nn.Conv2d(2*n_feats, n_feats, kernel_size=1, bias=True) 

        # stage 2
        self.conv_3_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.conv_5_2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=2, dilation=2, bias=True) # 7*7 conv

        self.confusion = nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        input_1 = x
        output_3_1 = self.conv_3_1(input_1) # 3*3 conv
        output_5_1 = self.conv_5_1(input_1) #3*3 conv
        input_2 = self.relu(torch.cat([output_3_1, output_5_1], 1))
        input_2 = self.fuse1(input_2) # 1*1 conv
        
        output_3_2 = self.conv_3_2(input_2)
        output_5_2 = self.conv_5_2(input_2)

        input_3 = self.relu(torch.cat([output_3_2, output_5_2], 1))
        output = self.confusion(input_3)

        output += x

        return output


class Edge_Net(nn.Module):
    def __init__(self, indim, hiddim, conv=default_conv, n_MSRB=3):
        
        super(Edge_Net, self).__init__()
        
        kernel_size = 3
        self.n_MSRB = n_MSRB 
       
        modules_head = [conv(2, hiddim, kernel_size)] #3*3 conv
        # body
        modules_body = nn.ModuleList()
        for i in range(n_MSRB):
            modules_body.append(MSRB(hiddim))

        # tail
        modules_tail = [nn.Conv2d(hiddim* (self.n_MSRB + 1), hiddim, 1, padding=0, stride=1), conv(hiddim, 1, kernel_size)]

        self.Edge_Net_head = nn.Sequential(*modules_head)
        self.Edge_Net_body = nn.Sequential(*modules_body)
        self.Edge_Net_tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        """
        x: (B, 2, H, W)
        """
        x = self.Edge_Net_head(x) #(B, hiddim, H, W)
        res = x

        MSRB_out = []
        for i in range(self.n_MSRB):
            x = self.Edge_Net_body[i](x)
            MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1) #(B, hiddim*(self.n_MSRB+1), H, W)
        x = self.Edge_Net_tail(res)

        return x


#============================== 

class attHead_image(nn.Module):
    """
    norm + 1*1 conv + 3*3 dconv
    """
    
    def __init__(self, C, C1, layernorm_type= 'BiasFree', bias=False):
        """
        C: input channel 
        C1: output channel after 1*1 conv

        """
        super(attHead_image,self).__init__()
        self.norm = LayerNorm(C, layernorm_type)
        self.conv1 = nn.Conv2d(C, C1, kernel_size=1, bias=bias) 
        self.conv2 = nn.Conv2d(C1, C1, kernel_size=3, stride=1, padding=1, groups=C1, bias=bias)

    def forward(self, x):
        """
        x: (B, C, H, W)
        """ 
        x1 = self.conv1(x) # (B, C1, H, W)
        x1 = self.conv2(x1)  #(B, C1, H, W)
        return x1



class EEM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias):
        """
        edge enhance module (EEM)
        C: input channel of image 
        C1: input channel of edge
        C2 : output channel of imhead/ehead
        """

        super(EEM, self).__init__()

        self.imhead = attHead_image(2, 2 * C2)
        self.ehead = nn.Conv2d(1, C2, kernel_size=3, stride=1, padding=1, groups=1, bias=bias)
        
        self.num_heads = num_heads
         
        self.a1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.project_out = nn.Conv2d(C2, 2, kernel_size=1, bias=bias)
        

    def edge_att(self, x, e):
        """
        edge attention
        x: input image (B, C, H, W)
        e: input edge (B, C1, H, W)
        """
        
        _, _, H, W = x.shape

        #=================================
        # high freq, edge
        # split into q, k, v 
        q1 = self.imhead(x) #(B, 2*C2, H, W) 
        k_eg = self.ehead(e) #(B, C2, H, W) 
        q_im, v_im = q1.chunk(2, dim=1)

        q_im = rearrange(q_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) 
        k_eg = rearrange(k_eg, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_im = rearrange(v_im, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #(B, head, C, H*W)

        q_im = torch.nn.functional.normalize(q_im, dim=-1)
        k_eg = torch.nn.functional.normalize(k_eg, dim=-1)
        
        attn = (q_im @ k_eg.transpose(-2, -1)) * self.a1 #(B, head, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v_im)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
       
        # skip connection
        out = x + self.project_out(out) #(B, 2, H, W)

        return out.contiguous()


    def forward(self, x, e):
        
        return self.edge_att(x,e)



class EAM(nn.Module):
    def __init__(self, C, C1, C2, num_heads, bias, shift=False):
        super(EAM, self).__init__()
        
        self.b1 = EEM(C, C1, C2, num_heads, bias) 
        self.dc = DC_multicoil(shift=shift)

    def forward(self, x, e, y, m, sens_map=None):

        xout = self.b1(x, e) #(B, 2, H, W)
        xout = self.dc(xout, y, m, sens_map) #(B, coils, H, W, 2)

        return xout


class EAMRI(nn.Module):
    """
    12 DAM + transformer block
    """
    def __init__(self, 
                indim=2, 
                edgeFeat=16, 
                attdim=4, 
                num_head=4, 
                fNums=[16,16,16,16,16],
                num_iters=[3,3,1,3,3], 
                n_MSRB=3, 
                shift=True): 
        """
        input:
            C: number of conv in RDB, default 3
            G0: input_channel, default 2
            G1: output_channel, default 2
            n_RDB: number of RDBs 

        """
        super(EAMRI, self).__init__()
       
        # sens_net 
        self.sens_net = SensitivityModel(chans = 8, num_pools = 4, shift=shift)

        # image module
        self.imHead = rdn_convBlock(inChannel=indim, midChannel=fNums[0], recursiveTime=num_iters[0], shift=shift)
        self.net1 = rdn_convBlock(inChannel=indim, midChannel=fNums[1], recursiveTime=num_iters[1], shift=shift)
        self.net2 = rdn_convBlock(inChannel=indim, midChannel=fNums[2], recursiveTime=num_iters[2], shift=shift)
        self.net3 = rdn_convBlock(inChannel=indim, midChannel=fNums[3], recursiveTime=num_iters[3], shift=shift)
        self.net4 = rdn_convBlock(inChannel=indim, midChannel=fNums[4], recursiveTime=num_iters[4], shift=shift)
        
        # edge module
        self.edgeNet = Edge_Net(indim=indim, hiddim=edgeFeat, n_MSRB=n_MSRB) # simple edge block

        # edgeatt module
        self.fuse1 = EAM(indim, 1, attdim, num_head, bias=True, shift=shift) 
        self.fuse2 = EAM(indim, 1, attdim, num_head, bias=True, shift=shift) 
        self.fuse3 = EAM(indim, 1, attdim, num_head, bias=True, shift=shift) 
        self.fuse4 = EAM(indim, 1, attdim, num_head, bias=True, shift=shift) 


    def reduce(self, x, sens_map):
        
        x1 = T.reduce_operator(x, sens_map, dim=1) #(B, H, W, 2)
        x1 = x1.permute(0,3,1,2).contiguous() #(B,2,H,W)
        return x1 


    def forward(self, x1, y, m): # (image, kspace, mask)
        """
        input:
            x1: (B, coils, H, W, 2) zero-filled image
            e0: (B, 2, H, W) edge of zim
            y: under-sampled kspace 
            m: mask

        """
        # estimated sens map 
        sens_map = self.sens_net(y, m)
       
        x1 = self.reduce(x1, sens_map) #(B, 2, H, W)
        
        # image head 
        x1 = self.imHead(x1, y, m, sens_map) #(B, 2, H, W)
        
        # first stage
        x2 = self.net1(x1, y, m, sens_map) #(B, 2, H, W)
        e2 = self.edgeNet(x1) #(B, 1, H, W)
        x1 = self.fuse1(x1, e2, y, m, sens_map)

        # second stage
        x2 = self.net2(x1, y, m, sens_map) #(B, 2, H, W)
        e3 = self.edgeNet(x1) 
        x1 = self.fuse2(x1, e3, y, m, sens_map)

        # third stage
        x2 = self.net3(x1, y, m, sens_map) #(B, 2, H, W)
        e4 = self.edgeNet(x1) 
        x1 = self.fuse3(x1, e4, y, m, sens_map)

        x2 = self.net4(x1, y, m, sens_map) #(B, 2, H, W)
        e5 = self.edgeNet(x1) 
        x1 = self.fuse4(x1, e5, y, m, sens_map)
        
        return [e2,e3,e4,e5,x1]   

class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.
    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
        shift:bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super(SensitivityModel, self).__init__()
        self.mask_center = mask_center
        self.shift = shift
        
        self.norm_unet = sensNet(convNum=3, recursiveTime=1, inChannel=2, midChannel=8) 

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape 

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, coils, H, W, 2)
        assert x.shape[-1] == 2, "the last dimension of input should be 2"
        tmp = T.root_sum_of_squares(T.root_sum_of_squares(x, dim=4), dim=1).unsqueeze(-1).unsqueeze(1)
        
        return x/tmp
        #return T.safe_divide(x, tmp).cuda()


    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            # mask: [B, 1, H, W, 1]
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad, num_low_frequencies_tensor

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:

        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = T.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        # images: [96, 1, 218, 179, 2]
        # batches: 8
        images, batches = self.chans_to_batch_dim(T.ifft2(masked_kspace, shift=self.shift))
        
        # estimate sensitivities
        return self.divide_root_sum_of_squares(self.batch_chans_to_chan_dim(self.norm_unet(images), batches))


class sensNet(nn.Module):
    def __init__(self, convNum = 3, recursiveTime = 3, inChannel = 24, midChannel=32):
        super(sensNet, self).__init__()
        self.rTime = recursiveTime
        self.LRelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(inChannel,midChannel,3,padding = 1)
        self.dilateBlock = dilatedConvBlock(convNum, midChannel)
        self.conv2 = nn.Conv2d(midChannel,inChannel,3,padding = 1)


    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


    def forward(self, x1):

        x1 = self.complex_to_chan_dim(x1) 

        x2 = self.conv1(x1)
        x2 = self.LRelu(x2)
        xt = x2
        for i in range(self.rTime):
            x3 = self.dilateBlock(xt)
            xt = x3+x2
        x4 = self.conv2(xt)
        x4 = self.LRelu(x4)
        x5 = x4+x1
        
        x5 = self.chan_complex_to_last_dim(x5)
        return x5

