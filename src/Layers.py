# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:35:59 2019

@author: remus
"""

import torch
from torch import nn
import numpy as np
# https://medium.com/aureliantactics/location-cnn-and-pygame-learning-environment-in-ray-f26c96245eb1
# https://pytorch.org/docs/master/notes/extending.html#extending-torch-nn
# https://discuss.pytorch.org/t/how-to-impelement-a-customized-convolutional-layer-conv2d/5799
# https://github.com/szagoruyko/diracnets/blob/master/diracconv.py
# https://discuss.pytorch.org/t/custom-a-new-convolution-layer-in-cnn/43682
class CoordConvBase(nn.Module):
    # Paras copied from conv3d()
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(CoordConvBase, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation= dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.CoordChs = None
        
class CoordConv2d(CoordConvBase):
    # Input should have shape [N, C, H, W]
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        
    def forward(self, input):
        N, C, H, W = np.shape(input)
        ys, xs = np.meshgrid(np.arange(0,H), np.arange(0,W), indexing='ij') # WHY?
        xs = np.repeat(xs[np.newaxis,np.newaxis , :, :], N, axis=0)
        ys = np.repeat(ys[np.newaxis,np.newaxis , :, :], N, axis=0)
        x = np.concatenate((input, xs, ys), axis = 1)
        x = nn.Conv2d(self.in_channels + 2, self.out_channels, self.kernel_size, self.stride,
                 self.padding, self.dilation, self.groups,
                 self.bias, self.padding_mode)(x)
        return x


class CoordConv3d(CoordConvBase):
    # Input should have shape [N, C, T, H, W]
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        
    def forward(self, input):
#        print(input.dtype)
#        if self.CoordChs == None:        
        N, C, T, H, W = np.shape(input)
        zs, ys, xs = torch.meshgrid([torch.arange(0,T), torch.arange(0,H), torch.arange(0,W)])
        xs = xs.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1, 1).float().to(input.device, dtype = torch.float)
        ys = ys.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1, 1).float().to(input.device, dtype = torch.float)
        zs = zs.unsqueeze(0).unsqueeze(0).repeat(N, 1, 1, 1, 1).float().to(input.device, dtype = torch.float)
#            self.CoordChs = torch.cat((xs,ys,zs), 1)
        x = torch.cat((input, xs, ys, zs), 1)
#        x = torch.cat((input, self.CoordChs), 1)
#        zs, ys, xs = np.meshgrid(np.arange(0,T), np.arange(0,H), np.arange(0,W), indexing='ij')
#        xs = np.repeat(xs[np.newaxis,np.newaxis , :, :], N, axis=0)
#        ys = np.repeat(ys[np.newaxis,np.newaxis , :, :], N, axis=0)
#        zs = np.repeat(zs[np.newaxis,np.newaxis , :, :], N, axis=0)
#        x = torch.from_numpy(np.concatenate((input, xs, ys, zs), axis = 1)).double()
        x = nn.Conv3d(self.in_channels + 3, self.out_channels, self.kernel_size, self.stride,
                 self.padding, self.dilation, self.groups,
                 self.bias, self.padding_mode).to(device = input.device)(x)
        return x

#%%
#def testF(**args):
#    print(*args)
#    
#testF(a=1, b=2)