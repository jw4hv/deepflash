#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:55:43 2019

@author: jrxing
"""


import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from fileIO.io import safeLoadMedicalImg
class DataSet2D(Dataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """
    def __init__(self, imgs,  transform = None, device = torch.device("cpu")):        
        super(DataSet2D, self).__init__()
        # img should be [N,H,W,C] or [N, T, H, W, C]
#        data = safeLoadMedicalImg(img_filename)
        data = imgs
#        data = data.astype(float)
        # Note that in transform pytorch assume the image is PILImage [H, W, C]!
        self.data = data
#        self.data = torch.from_numpy(data[:, np.newaxis, :, :])
        self.transform= transform
#        if transform != None:
#            self.data = self.transform(self.data)
#        self.data.to(dtype=torch.float)
        self.dataShape = self.data.shape
    
    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
#        sample = self.data[idx, :, :, :]
#        sample = self.data[idx:idx+1, :, :].astype(np.float32)
#        sample = self.data[:, :, idx:idx+1].astype(np.float32)
        sample = self.data[idx, :].astype(np.float32)
        if self.transform:
#            sample = transforms.ToPILImage()(sample)
            sample = self.transform(sample)     
        return sample


class DataSet3D(Dataset):
    """
    Load in 3D medical image, crop into patches
    """
    def __init__(self):
        super(AEDataSet3D, self).__init__()

class DataSetDeep(Dataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """
    def __init__(self, source_data, target_data, groundtruth,  transform = None, device = torch.device("cpu")):        
        super(DataSetDeep, self).__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.groundtruth = groundtruth
        self.transform= transform
        
    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.source_data)[0]

    
    def __getitem__(self, idx):
#        sample = self.data[idx, :, :, :]
#        sample = self.data[idx:idx+1, :, :].astype(np.float32)
#        sample = self.data[:, :, idx:idx+1].astype(np.float32)
        src_sample = self.source_data[idx, :].astype(np.float32)
        tar_sample = self.target_data[idx, :].astype(np.float32)
        gd_sample = self.groundtruth[idx,:].astype(np.float32)
    
        if self.transform:
#            sample = transforms.ToPILImage()(sample)
            src_sample = self.transform(src_sample)  
            tar_sample = self.transform(tar_sample)
            gd_sample = self.transform(gd_sample)

        sample = {'source': src_sample, 'target': tar_sample,'gtru': gd_sample}

        return sample

class DataSetDeepPred(Dataset):
    """
    Load in 3D medical source target for predicting
    """
    def __init__(self, source_data, target_data, transform = None, device = torch.device("cpu")):        
        super(DataSetDeepPred, self).__init__()
        self.source_data = source_data
        self.target_data = target_data
        self.transform= transform
        # self.dataShape = self.source_data.shape
        
    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.source_data)[0]

    
    def __getitem__(self, idx):
#        sample = self.data[idx, :, :, :]
#        sample = self.data[idx:idx+1, :, :].astype(np.float32)
#        sample = self.data[:, :, idx:idx+1].astype(np.float32)
        src_sample = self.source_data[idx, :].astype(np.float32)
        tar_sample = self.target_data[idx, :].astype(np.float32)
        
        if self.transform:
#            sample = transforms.ToPILImage()(sample)
            src_sample = self.transform(src_sample)  
            tar_sample = self.transform(tar_sample)

        sample = {'source': src_sample, 'target': tar_sample}

        return sample


