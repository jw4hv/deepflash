import torch
import sys, os, glob
import argparse
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from configs.getConfig import getConfig
from torch.utils.data import DataLoader
import SimpleITK as sitk
import os, glob
import numpy as np


def loadDataVol(inputfilepath, targetDim):
    SEG, COR, AXI = [0,1,2]
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

def runExp(config, srcreal, tarreal, velxreal, velyreal, velzreal, srcimag, tarimag, velximag, velyimag, velzimag):
    #1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #2. Set Data
    import numpy as np
    from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2
    #3. Load training data
    SEG, COR, AXI = [0,1,2]
    ##################Determine using 3D network or 2D network####
    if(config['net']['paras']['structure'] =='deepflash3D'):
        targetDim = 3 
    else:
        targetDim = 2
    ##################LOAD REAL NET DATA##########################
    input_src_data_R = loadDataVol(srcreal,targetDim)
    input_tar_data_R = loadDataVol(tarreal,targetDim)
    input_vel_data_x = loadDataVol(velxreal,targetDim)
    input_vel_data_y = loadDataVol(velyreal,targetDim)
    input_vel_data_z = loadDataVol(velzreal,targetDim)
    input_vel_data_R = np.concatenate((input_vel_data_x, input_vel_data_y,input_vel_data_z ), axis = targetDim+1)
    print (input_vel_data_R.shape)
    ##################LOAD IMAG NET DATA##########################
    input_src_data_I = loadDataVol(srcimag,targetDim)
    input_tar_data_I = loadDataVol(tarimag,targetDim)
    input_vel_data_x_I = loadDataVol(velximag,targetDim)
    input_vel_data_y_I = loadDataVol(velyimag,targetDim)
    input_vel_data_z_I = loadDataVol(velzimag,targetDim)
    input_vel_data_I = np.concatenate((input_vel_data_x_I, input_vel_data_y_I,input_vel_data_z_I ), axis = targetDim+1)
    print ('Training Data loaded!')
    from torchvision import transforms
    from fileIO.io import safeDivide
    #4. Add transformation
    xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) 
    trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
    img_transform = transforms.Compose([
        xNorm,
        trans3DTF2Torch
    #    transforms.ToTensor()
    ])
    from src.DataSet import DataSet2D, DataSetDeep, DataSetDeepPred
    training = DataSetDeep (source_data_R = input_src_data_R, target_data_R = input_tar_data_R,  groundtruth_R = input_vel_data_R, source_data_I = input_src_data_I, target_data_I = input_tar_data_I,  groundtruth_I = input_vel_data_I, transform=img_transform, device = device  )
    testing = DataSetDeep(source_data_R = input_src_data_R, target_data_R = input_tar_data_R,  groundtruth_R = input_vel_data_R, source_data_I = input_src_data_I, target_data_I = input_tar_data_I,  groundtruth_I = input_vel_data_I, transform=img_transform, device = device )
    from src.DFModel import DFModel
    #4. Weight initilization
    def weights_init_uniform(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)   
    #5. Load Traniners and testing model 
    deepflashnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    #6. Training and Validation
    loss = deepflashnet.trainDeepFlash(training_dataset=training, training_config = config['training'], valid_img= None, expPath = None)
    #7. check point file saving
    model_save_path = './save_trained_model/'
    if not os.path.exists(model_save_path ):
        os.makedirs(model_save_path)
    path = model_save_path + 'saved_model.pth'
    deepflashnet.save(path)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_type', type=str, help="choosing 2D or 3D network")

    parser.add_argument('--im_src_realpart', type=str, help="root directory of real parts of source images")
    parser.add_argument('--im_tar_realpart', type=str, help="root directory of real parts of target images")
    parser.add_argument('--im_vel_realX', type=str, help="root directory of real parts of velocity fields (X direction)")
    parser.add_argument('--im_vel_realY', type=str, help="root directory of real parts of velocity fields (Y direction)")
    parser.add_argument('--im_vel_realZ', type=str, help="root directory of real parts of velocity fields (Z direction)")

    parser.add_argument('--im_src_imaginarypart', type=str, help="root directory of imaginary parts of source images")
    parser.add_argument('--im_tar_imaginarypart', type=str, help="root directory of imaginary parts of target images")
    parser.add_argument('--im_vel_imagX', type=str, help="root directory of imaginary parts of source images (X direction)")
    parser.add_argument('--im_vel_imagY', type=str, help="root directory of imaginary parts of velocity fields (Y direction)")
    parser.add_argument('--im_vel_imagZ', type=str, help="root directory of imaginary parts of velocity fields (Z direction)")
    args,unknown= parser.parse_known_args()
    configName = args.network_type
    config = getConfig(configName)
    runExp(config, args.im_src_realpart, args.im_tar_realpart, args.im_vel_realX, args.im_vel_realY,args.im_vel_realZ, \
        args.im_src_imaginarypart, args.im_tar_imaginarypart, args.im_vel_imagX, args.im_vel_imagY,args.im_vel_imagZ)


