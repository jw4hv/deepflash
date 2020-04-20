#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 09:54:25 2019

@author: Javier
"""
from configs.getConfig import getConfig
import torch

def runExp(config, configName, resultPath = '../result', continueTraining = False, oldExpPath = None, addDate = True):
    #%% 1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import numpy as np
    from utils.io import safeLoadMedicalImg, convertTensorformat, loadData

    # Load training and valid data
    SEG, COR, AXI = [0,1,2]
    targetDim = 3
    training_data = loadData(config['data']['training'], config['data'], targetDim = targetDim, sourceSliceDim=AXI)
    valid_data = convertTensorformat(img=safeLoadMedicalImg(config['data']['valid'][0]),
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'pytorch', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)
    