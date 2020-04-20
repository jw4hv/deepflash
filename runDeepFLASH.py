import torch
import sys
from configs.getConfig import getConfig
from fileIO.io import saveConfig2Json
from fileIO.io import createExpFolder
from torch.utils.data import DataLoader
import SimpleITK as sitk





def runExp(config):
    #%% 1. Set configration and device    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create experiment result folder    
    # expPath, expName = createExpFolder(resultPath, configName, create_subFolder=True, addDate = addDate)
    # saveConfig2Json(config, expPath + '/config.json')

    #%% 2. Set Data
    import numpy as np
    from fileIO.io import safeLoadMedicalImg, convertTensorformat, loadData2

    # Load training and valid data
    SEG, COR, AXI = [0,1,2]
    targetDim = 3
    import os, glob
    #os.chdir("./data/Spatial") #Spatial data path
    os.chdir("./data/")   # Fourier Data path
    for idx, filename in enumerate (sorted(glob.glob("SourceBrain/*_src.mhd"), key=os.path.getmtime)):
        
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        
        temp = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)

        
        if idx == 0:        
            input_src_data = temp
        else:
            input_src_data = np.concatenate((input_src_data, temp), axis=0)
        print(np.shape(input_src_data[0]))
        # outfile = './%d.mhd' %idx
        # im = sitk.GetImageFromArray(input_src_data[idx], isVector=False)
        # sitk.WriteImage(im, outfile, True) 
    for idx, filename in enumerate (sorted(glob.glob("TargetBrain/*_tar.mhd"), key=os.path.getmtime)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        temp2 = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)
        if idx == 0:        
            input_tar_data = temp2
        else:
            input_tar_data = np.concatenate((input_tar_data, temp2), axis=0)
     

    # for idx, filename in enumerate (glob.glob("Fourier/Synthetic/VelocityFourierRealX/*.mhd")):
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    #     img = np.rollaxis(img, 0, 3)
    #     temp = convertTensorformat(img,
    #                             sourceFormat = 'single3DGrayscale', 
    #                             targetFormat = 'tensorflow', 
    #                             targetDim = targetDim, 
    #                             sourceSliceDim = AXI)
    #     if idx == 0:        
    #         input_vel_data = temp
    #     else:
    #         input_vel_data = np.concatenate((input_vel_data, temp), axis=0)
    filename="./3Dbrainlabel.txt"
    data = np.loadtxt(filename, delimiter=" ", 
                  usecols=[0])
    print(len(data))
    for idx in range (0, len(data)):
        img = data[idx]
        img = img.reshape(1,1,1,1,1)
        if idx == 0:        
            input_vel_data = img
        else:
            input_vel_data = np.concatenate((input_vel_data, img), axis=0)
    # input_vel_data = torch.from_numpy(data).float().to(device)
    # print (input_vel_data)


    from torchvision import transforms
    from fileIO.io import safeDivide

    xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) - 0.5
    trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
    img_transform = transforms.Compose([
        xNorm,
        trans3DTF2Torch
    #    transforms.ToTensor()
    ])
    from src.DataSet import DataSet2D, DataSetDeep, DataSetDeepPred
    training = DataSetDeep (source_data = input_src_data, target_data = input_tar_data,  groundtruth = input_vel_data, transform=img_transform, device = device  )


    from src.DFModel import DFModel
    def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

    
    deepflashnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    deepflashtestnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    #deepflashnet.apply(weights_init_uniform)

    for idx, filename in enumerate (sorted(glob.glob("SourceBrain/*_src.mhd"), key=os.path.getmtime)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        temp = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)
        if idx == 0:        
            pred_src_data = temp
        else:
            pred_src_data = np.concatenate((pred_src_data, temp), axis=0)

    for idx, filename in enumerate (sorted(glob.glob("TargetBrain/*_tar.mhd"), key=os.path.getmtime)):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        img = np.rollaxis(img, 0, 3)
        temp = convertTensorformat(img,
                                sourceFormat = 'single3DGrayscale', 
                                targetFormat = 'tensorflow', 
                                targetDim = targetDim, 
                                sourceSliceDim = AXI)
        if idx == 0:        
            pred_tar_data = temp
        else:
            pred_tar_data = np.concatenate((pred_tar_data, temp), axis=0)
        # outfile = './%d.mhd' %idx
        # im = sitk.GetImageFromArray(pred_src_data[0], isVector=False)
        # sitk.WriteImage(im, outfile, True) 
       
    testing = DataSetDeepPred (source_data = pred_src_data, target_data = pred_tar_data, transform=img_transform, device = device)




    #%% 6. Training
    loss = deepflashnet.trainDeepFlash(training_dataset=training, training_config = config['training'], testing_dataset = testing, valid_img= None, expPath = None)
    deepflashnet.save('./savenet3d.pth')
    # torch.save(deepflashnet.state_dict(), '2Dnet')
      
    deepflashtestnet = DFModel(net_config = config['net'], 
                        loss_config = config['loss'],
                        device=device)
    deepflashtestnet.load('./savenet.pth')
      # 7. Prediction 
    # for idx, filename in enumerate (sorted(glob.glob("Source2DTest/*.mhd"), key=os.path.getmtime)):
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    #     img = np.rollaxis(img, 0, 3)
    #     temp3 = convertTensorformat(img,
    #                             sourceFormat = 'single3DGrayscale', 
    #                             targetFormat = 'tensorflow', 
    #                             targetDim = targetDim, 
    #                             sourceSliceDim = AXI)
     
    #     if idx == 0:        
    #         pred_src_data = temp3
    #     else:
    #         pred_src_data = np.concatenate((pred_src_data, temp3), axis=0)

    #     # outfile = './%d.mhd' %idx
    #     # im = sitk.GetImageFromArray(pred_src_data[0], isVector=False)
    #     # sitk.WriteImage(im, outfile, True) 

    # for idx, filename in enumerate (sorted(glob.glob("Target2DTest/*.mhd"), key=os.path.getmtime)):
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
    #     img = np.rollaxis(img, 0, 3)
    #     temp4 = convertTensorformat(img,
    #                             sourceFormat = 'single3DGrayscale', 
    #                             targetFormat = 'tensorflow', 
    #                             targetDim = targetDim, 
    #                             sourceSliceDim = AXI)
    #     if idx == 0:        
    #         pred_tar_data = temp4
    #     else:
    #         pred_tar_data = np.concatenate((pred_tar_data, temp4), axis=0)

    # predfeed = DataSetDeepPred (source_data = pred_src_data, target_data = pred_tar_data, transform=img_transform, device = device)
    # predictions = deepflashnet.pred(predfeed, np.max(data))
    # predictions_file = predictions.reshape(201,1)
    # np.savetxt('a.txt',predictions_file,fmt='%f')
    # print (predictions[2])
    #loss_history = autoencoder.train(training_dataset=training_dataset, training_config = config['training'], valid_img= None, expPath = None)


configName = 'deepflash'
config = getConfig(configName)
runExp(config)


#     valid_data = convertTensorformat(img=safeLoadMedicalImg(config['data']['valid'][0]),
#                                 sourceFormat = 'single3DGrayscale', 
#                                 targetFormat = 'pytorch', 
#                                 targetDim = targetDim, 
#                                 sourceSliceDim = AXI)
    
#     #%% 3. Set transformations
#     from torchvision import transforms
#     from utils.io import safeDivide
#     #xNorm = lambda img : (img - np.min(img)) / (np.max(img) - np.min(img)) - 0.5
#     xNorm = lambda img : safeDivide(img - np.min(img), (np.max(img) - np.min(img))) - 0.5
#     trans3DTF2Torch = lambda img: np.moveaxis(img, -1, 0)
#     img_transform = transforms.Compose([
#         xNorm,
#         trans3DTF2Torch
#     #    transforms.ToTensor()
#     ])

#     #%% 4. Set dataset


#     training_dataset = DataSet2D(imgs = training_data, transform=img_transform, device = device)

#     #%% 5. Set network
#     from modules.AEModel import AEModel
#     autoencoder = AEModel(net_config = config['net'], 
#                         loss_config = config['loss'], 
#                         device=device)
#     if continueTraining:
#         # autoencoder.load(oldExpPath + '/model/model.pth')
#         autoencoder.load(oldExpPath + '/checkpoint/checkpoint.pth')

#     #%% 6. Training
#     print('Start configure: ' + expName)
#     loss_history, past_time = autoencoder.train(training_dataset=training_dataset, training_config = config['training'], valid_img=valid_data, expPath = expPath)
#     autoencoder.save(expPath + '/checkpoint/checkpoint.pth')

#     # Add loss and training time to config json file
#     config['training']['loss'] = loss_history[-1]
#     config['training']['time_hour'] = past_time / 3600
#     saveConfig2Json(config, expPath + '/config.json')

#     return expPath, expName, loss_history, past_time

# def runExpGroup(configGroup, configGruopName, resultPath = '../result', continueTraining = False, oldExpPath = None):    
#     expGroupPath, _ = createExpFolder(resultPath, configGruopName)
#     # resultPath += '/' + expGroupName
#     for expIdx, expConfig in enumerate(configGroup):
#         runExp(expConfig, f'idx-{expIdx}', expGroupPath, addDate = False)