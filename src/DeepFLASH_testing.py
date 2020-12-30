import torch
import sys, os, glob
import argparse
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from configs.getConfig import getConfig
from fileIO.io import saveConfig2Json
from fileIO.io import createExpFolder
from torch.utils.data import DataLoader
import SimpleITK as sitk
import os, glob
import numpy as np
from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2

def loadDataVol(inputfilepath):
    SEG, COR, AXI = [0,1,2]
    targetDim = 2
    for idx, filename in enumerate (sorted(glob.glob(inputfilepath), key=os.path.getmtime)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        temp = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)      
        if idx == 0:        
            outvol = temp
        else:
            outvol  = np.concatenate((outvol , temp), axis=0)
    return outvol

def runExp(config, savemodel, srcreal, tarreal, srcimag, tarimag):
    #1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #2. Set Data
    import numpy as np
    from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2
    #3. Load testing data
    SEG, COR, AXI = [0,1,2]
    targetDim = 2
    ##################LOAD REAL NET DATA##########################
    input_src_data_R = loadDataVol(srcreal)
    input_tar_data_R = loadDataVol(tarreal)
    ##################LOAD IMAG NET DATA##########################
    input_src_data_I = loadDataVol(srcimag)
    input_tar_data_I = loadDataVol(tarimag)
    from torchvision import transforms
    from fileIO.io import safeDivide
    #4. Add transformation
    xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) 
    trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
    img_transform = transforms.Compose([
        trans3DTF2Torch
    #    transforms.ToTensor()
    ])
    from src.DataSet import DataSetDeepPred
    testing = DataSetDeepPred(source_data_R = input_src_data_R, target_data_R = input_tar_data_R, source_data_I = input_src_data_I, target_data_I = input_tar_data_I, transform=img_transform, device = device )
    from src.DFModel import DFModel
    #5. Load network
    deepflashnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    #%% 6. Load trained model   
    deepflashnet.load(savemodel)
    # #%% 7. Testing
    predictions = deepflashnet.pred(dataset= testing, scale = 1)
if __name__ == "__main__":

    configName = 'deepflash'
    config = getConfig(configName)
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', type=str, help="root directory of trained model")
    parser.add_argument('--im_src_realpart', type=str, help="root directory of real parts of source images")
    parser.add_argument('--im_tar_realpart', type=str, help="root directory of real parts of target images")
    parser.add_argument('--im_src_imaginarypart', type=str, help="root directory of imaginary parts of source images")
    parser.add_argument('--im_tar_imaginarypart', type=str, help="root directory of imaginary parts of target images")
    
    args,unknown= parser.parse_known_args()
    runExp(config, args.saved_model, args.im_src_realpart, args.im_tar_realpart, \
        args.im_src_imaginarypart, args.im_tar_imaginarypart)


