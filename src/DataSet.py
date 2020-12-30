#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
        self.dataShape = self.data.shape
    
    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.data)[0]
    
    def __getitem__(self, idx):
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
    def __init__(self, source_data_R, target_data_R, groundtruth_R, source_data_I, target_data_I, groundtruth_I,  transform = None, device = torch.device("cpu")):        
        super(DataSetDeep, self).__init__()
        self.source_data_R = source_data_R
        self.target_data_R= target_data_R
        self.groundtruth_R = groundtruth_R

        self.source_data_I = source_data_I
        self.target_data_I = target_data_I
        self.groundtruth_I = groundtruth_I
        self.transform= transform
        
    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.source_data_R)[0]

    
    def __getitem__(self, idx):
        src_sample_R = self.source_data_R[idx, :].astype(np.float32)
        tar_sample_R = self.target_data_R[idx, :].astype(np.float32)
        gd_sample_R = self.groundtruth_R[idx,:].astype(np.float32)
        
        src_sample_I = self.source_data_I[idx, :].astype(np.float32)
        tar_sample_I = self.target_data_I[idx, :].astype(np.float32)
        gd_sample_I = self.groundtruth_I[idx,:].astype(np.float32)
        if self.transform:
            src_sample_R = self.transform(src_sample_R)  
            tar_sample_R = self.transform(tar_sample_R)
            gd_sample_R = self.transform(gd_sample_R)
            src_sample_I = self.transform(src_sample_I)  
            tar_sample_I = self.transform(tar_sample_I)
            gd_sample_I = self.transform(gd_sample_I)

        sample = {'source_R': src_sample_R, 'target_R': tar_sample_R,'gtru_R': gd_sample_R, 'source_I': src_sample_I, 'target_I': tar_sample_I,'gtru_I': gd_sample_I}

        return sample

class DataSetDeepPred(Dataset):
    """
    Load in 3D medical source target for predicting
    """
    def __init__(self, source_data_R, target_data_R, source_data_I, target_data_I,transform = None, device = torch.device("cpu")):        
        super(DataSetDeepPred, self).__init__()
        self.source_data_R = source_data_R
        self.target_data_R= target_data_R
        self.source_data_I = source_data_I
        self.target_data_I = target_data_I
        self.transform= transform

    def __len__(self):
#        return np.shape(self.data)[2]
        return np.shape(self.source_data_R)[0]

    
    def __getitem__(self, idx):
        src_sample_R = self.source_data_R[idx, :].astype(np.float32)
        tar_sample_R = self.target_data_R[idx, :].astype(np.float32)
        src_sample_I = self.source_data_I[idx, :].astype(np.float32)
        tar_sample_I = self.target_data_I[idx, :].astype(np.float32)
        
        if self.transform:
            src_sample_R = self.transform(src_sample_R)  
            tar_sample_R = self.transform(tar_sample_R)
            src_sample_I = self.transform(src_sample_I)  
            tar_sample_I = self.transform(tar_sample_I)
            

        sample = {'source_R': src_sample_R, 'target_R': tar_sample_R,'source_I': src_sample_I, 'target_I': tar_sample_I}

        return sample


