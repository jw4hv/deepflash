# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 09:11:33 2019

"""
import numpy as np

def safeLoadMedicalImg(filename, read_dim = 0):
    import nibabel as nib
    import SimpleITK as sitk
    dataFormat = filename.split('.')[-1]
    if dataFormat.lower() == 'nii':
        img = nib.load(filename).get_fdata()
    elif dataFormat.lower() == 'mhd':
        # Strangely, [80, 70, 50] image become [50, 80, 70]!
        # i.e. [H, W, N] => [N, H, W]
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
    
    if read_dim != 2:
        img = np.rollaxis(img, read_dim, 2)
            
    return img


def convertTensorformat(img, sourceFormat, targetFormat, targetDim=0, sourceSliceDim=0):
    '''
    @description:
        Convert ime tensor format among 3D medical images, Tensorflow and Pytorch
    '''
    # 1. Convert from source format to tensorflow in same dimension
    if sourceFormat.lower() == 'single3dgrayscale':
        # [D1, D2, D3] -> [1, D, H, W, 1]
        img = np.rollaxis(img, sourceSliceDim)
        img = img[np.newaxis, :, :, :, np.newaxis]
        # img = np.expand_dims(img, -1)
    elif sourceFormat.lower() == 'tensorflow':
        pass
    elif sourceFormat.lower() == 'pytorch':
        img =  np.moveaxis(img, 1, -1)
#        img = np.moveaxis(img, -1, 1)
    
    sourceDim = len(np.shape(img)) - 2
    if targetDim == 0:
        targetDim = sourceDim
        
    # 2. Convert to target format
    if targetFormat != 'tensorflow':
        return convertTFTensorTo(img, targetFormat, targetDim)
    else:        
        if sourceDim == targetDim:
            return img
        elif sourceDim == 2:
            # [N, H, W, C] -> [N, D, H, W, C]
            return img[:,np.newaxis,:,:,:]
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [N, H, W, C]            
            if np.shape(img)[1] != 1:
                raise ValueError(f'Cannot convert')
            else:
                return img[:, 0, :, :, :]

def convertTFTensorTo(img, targetFormat, targetDim):
    '''    
    @description: 
        Convert tensorflow tensor format into 3D grayscale or Pyrotch format tensors
    @params:
        img{numpy ndarray}:
            tensorflow tensor with shape [N, H, W, C] or [N, D, H, W, C]
        lang{string}:
            single3DGrayscale or Pyrotch.
            single3DGrayscale:
                2D: [D, H, W]
            Pyrotch:
                2D: [N, C, H, W]
                3D: [N, C, T, H, W]
        dim{int}:
            Treate the 3D image as stack of 2D or 3D tensors
    @return:         
    '''
    sourceDim = len(np.shape(img)) - 2
    
    if targetFormat.lower() == 'single3dgrayscale':
        # [N, H, W, C] or [N, D, H, W, C] -> [D, H, W]        
        if sourceDim == 2:
            # [N, H, W, C] -> ?
            raise ValueError('Not supported')
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [D, H, W]
            if np.shape(img)[0] == 1 and np.shape(img)[-1] == 1:
                return img[0,:,:,:,0]
            else:
                raise ValueError('Not supported')
            
    elif targetFormat.lower() == 'pytorch':
        if sourceDim == targetDim:
            # [N, D, H, W, C] -> [N, C, D, H, W]
            # [N, H, W, C] -> [N, C, H, W]
            return np.moveaxis(img, -1, 1)
        elif sourceDim == 2:
            # [N, H, W, C] -> [N, C, H, W] -> [N, C, D, H, W]
            img = np.moveaxis(img, -1, 1)
            return img[:,:,np.newaxis,:,:]
        elif sourceDim == 3:
            # [N, D, H, W, C] -> [N, H, W, C] -> [N, C, H, W]
            if np.shape(img)[1] != 0:
                raise ValueError('Not supported')
            else:
                img = img[:,0,:,:]
                return np.moveaxis(img, -1, 1)
    
def convertMedicalTensorTo(img, lang, dim, sliceDim = 0):
    '''    
    @description: 
        Convert 3D medical image (with dim X, Y, Z) into Tensorflow or Pyrotch format tensors
    @params:
        img{numpy ndarray}:
            3D medical image with shape [X, Y, Z]
        lang{string}:
            Tensorflow or Pyrotch.
            Tensorflow:
                2D: [N, H, W, C]
                3D: [N, D, H, W, C]
            Pyrotch:
                2D: [N, C, H, W]
                3D: [N, C, T, H, W]
        dim{int}:
            Treate the 3D image as stack of 2D or 3D tensors
        sliceDim{int}:
            if dim == 2, then sliceDim refer to teh dim to look into (slice)
    @return:         
    '''
    # 1. Convert image into [N, H, W]
    img = np.rollaxis(img, sliceDim)
    
    # 2. Convert to target format
    if dim == 2 and lang.lower() == 'tensorflow':
        img = img[:, :, :, np.newaxis]
    elif dim == 2 and lang.lower() == 'pytorch':
        img = img[:, np.newaxis, :, :]
    elif dim == 3 and lang.lower() == 'tensorflow':
        img = img[np.newaxis, :, :, :, np.newaxis]
    elif dim == 3 and lang.lower() == 'pytorch':
        img = img[:, np.newaxis, np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported combination: target format {lang} with dim {dim}")
    return img

def safeDivide(a,b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def createExpFolder(resultPath, configName, create_subFolder = False, addDate = True):
    # https://datatofish.com/python-current-date/
    import os
    import datetime
    if addDate:
        today = datetime.date.today().strftime('%d-%b-%Y')
        expName = today + '-' + configName
    else:
        expName = configName
    expPath = resultPath + '/' + expName
    if not os.path.exists(expPath):
        os.mkdir(expPath)
    else:
        for idx in range(2,100):
            if addDate:
                expName = today + '-' + configName + '-' + str(idx)
            else:
                expName = configName + '-' + str(idx)
            expPath = resultPath + '/' + expName
            if not os.path.exists(expPath):
                os.mkdir(expPath)
                break
    if create_subFolder:
        os.mkdir(expPath + '/valid_img')
        os.mkdir(expPath + '/train_img')
        os.mkdir(expPath + '/checkpoint')

    return expPath, expName


# def setExp(expName, resultPath='../result', continueTraining = False):
#     import json
#     from configs.getConfig import getConfig
#     if continueTraining:
#         oldExpName = expName
#         oldExpPath = resultPath + '/' + oldExpName
#         expName = oldExpName + '-Continue'
#         with open(oldExpPath + '/config.json', 'r') as f:
#             config = json.load(f)
#     else:
#         # expName = 'debug'
#         # expName = 'default'
#         config = getConfig(name=expName)
#     return expName, config
def getConfigFromFile(jsonFilename):
    import json
    with open(jsonFilename, 'r') as f:
        config = json.load(f)
    return config

def saveConfig2Json(config, jsonFilename):
    import json
    with open(jsonFilename, 'w') as f:
        json.dump(config, f)


def loadData(data_filenames, data_config,
             sourceFormat = 'single3DGrayscale', targetFormat = 'tensorflow', 
                               targetDim = 3, sourceSliceDim = 0):
    for idx, data_filename in enumerate(data_filenames):
        datum = convertTensorformat(img=safeLoadMedicalImg(data_filename),
                               sourceFormat = sourceFormat, 
                               targetFormat = targetFormat, 
                               targetDim = targetDim, 
                               sourceSliceDim = sourceSliceDim)        
        if idx == 0:        
            data = datum
        else:
            data = np.concatenate((data, datum), axis=0)
    return data

def loadData2(data_filenames,sourceFormat = 'single3DGrayscale', targetFormat = 'tensorflow', 
                               targetDim = 3, sourceSliceDim = 0):
    for idx, data_filename in enumerate(data_filenames):
        datum = convertTensorformat(img=safeLoadMedicalImg(data_filename),
                               sourceFormat = sourceFormat, 
                               targetFormat = targetFormat, 
                               targetDim = targetDim, 
                               sourceSliceDim = sourceSliceDim)

    return datum

def loadData3(data_filename,sourceFormat = 'single3DGrayscale', targetFormat = 'tensorflow', 
                               targetDim = 3, sourceSliceDim = 0):
    
    datum = convertTensorformat(img=safeLoadMedicalImg(data_filename),
                               sourceFormat = sourceFormat, 
                               targetFormat = targetFormat, 
                               targetDim = targetDim, 
                               sourceSliceDim = sourceSliceDim)

    return datum
