#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
def getDFNET(config):
    net_type = config.get('type', 'AECV2D')
    if net_type == 'AECV2D':
        return AECV2D(config)
    elif net_type == 'AECV3D':
        return AECV3D(config)
    elif net_type == 'DeepCC':
        return DeepCC(config)  
    elif net_type == 'DF':
        return DeepFlash(config)  
    else:
        raise ValueError(f'Unsupported network type: {net_type}')



class AECV3D(nn.Module):
    def __init__(self, config):
        super(AECV3D, self).__init__()
        self.imgDim = 3
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepCC(nn.Module):
    def __init__(self, config):
        super(DeepCC, self).__init__()
        self.imgDim = 3
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DeepFlash(nn.Module):
    def __init__(self, config):
        super(DeepFlash, self).__init__()
        self.imgDim = 2
        paras = config.get('paras', None)
        self.encoder, self.decoder = get3DNet(paras)
    
    def forward(self, src, tar):
        x1 = self.encoder(src)
        x2 = self.encoder(tar)
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy() 
        x3 = np.concatenate((x1, x2), axis=1)
        x3 = torch.Tensor (x3)
        x3 = self.decoder(x3)
        # x3 = 12 *x3
        return x3    
def get3DNet(paras):
    if paras == None or paras['structure'] == 'default':        
        encoder = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.Conv3d(8, 16, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # 
            nn.Conv3d(16, 32, 3, stride=1, padding=1),  # 
            nn.ReLU(True),
            nn.Conv3d(32, 32, 3, stride=1, padding=1),  # 
            nn.ReLU(True),

        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Conv3d(16, 8, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )

    elif paras['structure'] == 'complex':
        encoder = nn.Sequential(
            nn.Conv3d(2, 8, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool3d(2, stride=2),  # b, 16, 5, 5
            nn.Conv3d(8, 16, 3, stride=2, padding=1),  # b, 8, 3, 3            
            nn.ReLU(True)
        )
        decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 2, stride=2, padding=0),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.Tanh(),
        )
    elif paras['structure'] == 'deepflash3D':
       
###########################3D Net (LEARNING RATE = 1E-2)##########################################
        encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, stride=2, padding=2, bias = True),  # Large kernel size, large output feature map, and with dense stride
            nn.BatchNorm3d(16),
            nn.ReLU(),
           
            nn.Conv3d(16, 16, 3, stride=2, padding=1, bias = True), 
            nn.BatchNorm3d(16),
        nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Conv3d(16, 32, 3, stride=1, padding=2, bias = True), 
            nn.BatchNorm3d(32),
        nn.ReLU(), 
            nn.Conv3d(32, 64, 3, stride=2, padding=2,bias = True), 
            nn.BatchNorm3d(64),
        nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Conv3d(64, 128, 3, stride=2, padding=2, bias = True), 
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        decoder = nn.Sequential(
            nn.Conv3d(256, 128, 3, stride=1, padding=2, bias = True),  # 
            nn.BatchNorm3d(128),
           

            nn.Conv3d(128, 64, 3, stride=1, padding=2, bias = True),  # 
            nn.BatchNorm3d(64),
            nn.Dropout(0.2), 
        nn.ReLU(),
            nn.Conv3d(64, 32, 3, stride=1, padding=2, bias = True), 
            nn.BatchNorm3d(32),
        nn.ReLU(),

            nn.Conv3d(32, 8, 3, stride=1, padding=1, bias = True), 
            nn.BatchNorm3d(8),
        nn.ReLU(),
            nn.Conv3d(8, 3, 3, stride=1, padding=1,bias = True), 
        )
    elif paras['structure'] == 'deepflash':
##########################2D Net#################################################
        encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1,bias = True),  #   Loss = MSE ; 'batch_size'] = 256 'learning_rate' = 5e-3
            nn.Dropout(0.2), 
            nn.BatchNorm2d(8),
            nn.PReLU(),
         
            nn.Conv2d(8, 16, 3, stride=1, padding=2,bias = True),  # 
            nn.BatchNorm2d(16),
            nn.PReLU(),
                   
            nn.Conv2d(16, 8, 3, stride=1, padding=2,bias = True),  # 
            nn.BatchNorm2d(8),
            nn.PReLU(),

            nn.Conv2d(8, 1, 4, stride=2, padding=2,bias = True),  # 
            nn.Dropout(0.2), 
            nn.BatchNorm2d(1),
            nn.PReLU(),
    
        )
        decoder = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=1, padding=1,bias = True),
            nn.BatchNorm2d(8), 
            nn.PReLU(),

            nn.Conv2d(8, 16, 5, stride=1, padding=1,bias = True),
            nn.BatchNorm2d(16),
            nn.PReLU(),

            nn.Conv2d(16, 8, 3, stride=1, padding=1,bias = True),
            nn.MaxPool2d(3), 
            nn.BatchNorm2d(8),
            nn.Dropout(0.2),
            nn.PReLU(),

            nn.Conv2d(8, 3, 3, stride=1, padding=1,bias = True),  
        )

    return encoder, decoder


from src.Layers import CoordConv3d
#from Layers import CoordConv3d