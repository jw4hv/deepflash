#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import torch
from src.DFNet import getDFNET
from torch.utils.data import DataLoader
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn as nn

class DFModel(object):
    def __init__(self, net_config, loss_config, device = torch.device("cpu")):
        # Set network structure
        self.device = device
        self.net = getDFNET(net_config)
        self.net.to(device)
        self.criterion = get_loss(loss_config)
        self.continueTraining = False
    
    
    #def train(self, training_dataset, training_config, groundtruth_dataset, valid_img = None, expPath = None):
    def train(self, training_dataset, training_config, valid_img = None, expPath = None):
        # Create dataloader
        training_dataloader = DataLoader(training_dataset, batch_size=training_config['batch_size'],
                        shuffle=True)
        # Set Optimizer
        if self.continueTraining == False:        
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=training_config['learning_rate'],
                                weight_decay=10e-6)

        # Save valid image if needed
        ifValid = training_config.get('valid_check', False) and valid_img is not None
        if ifValid:
            prtDigitalLen = len(str(training_config["epochs_num"]))
            valid_truth = slice_img(valid_img, training_config['valid_check'])
            plt.imsave(expPath + '/valid_img/valid_epoch_' + '0'.zfill(prtDigitalLen) + '.png', np.squeeze(valid_truth), cmap='gray')
        


        # Training process
        start_time = time.time()
        loss_history = np.zeros([0])
        print("Training started") 
        for epoch in range(1, training_config['epochs_num']  + 1):
            for i, data in enumerate(training_dataloader):      
                img = data['input'].to(self.device, dtype = torch.float)
                img1 = data['input'].to(self.device, dtype = torch.float)
                label = data['gtru'].to(self.device, dtype = torch.float)
                # ===================forward=====================
                output = self.net(img)
                loss = self.criterion(output, label)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss)
            # ===================log========================            
            report_epochs_num = training_config.get('report_per_epochs', 10)            
        print('Training finished')
        return loss

    def trainDeepFlash(self, training_dataset, training_config, valid_img = None, expPath = None):
        # Create dataloader
        training_dataloader = DataLoader(training_dataset, batch_size=training_config['batch_size'],
                        shuffle=False)

        # Set Optimizer
        if self.continueTraining == False:        
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=training_config['learning_rate'],
                                weight_decay=1e-6)
        TotalEnergy = 0
        print("Training started") 
        for j, epoch in enumerate(range(1, training_config['epochs_num']  + 1)):
            for i, data in enumerate(training_dataloader):      
                img_src_R = data['source_R'].to(self.device, dtype = torch.float)
                img_tar_R = data['target_R'].to(self.device, dtype = torch.float)
                label_R = data['gtru_R'].to(self.device, dtype = torch.float)

                img_src_I = data['source_I'].to(self.device, dtype = torch.float)
                img_tar_I = data['target_I'].to(self.device, dtype = torch.float)
                label_I = data['gtru_I'].to(self.device, dtype = torch.float)
                # ===================forward=====================
                output_R = self.net(img_src_R,img_tar_R)
                output_I = self.net(img_src_I,img_tar_I)
                lossR = self.criterion(output_R, label_R)
                lossI = self.criterion(output_I, label_I)
                loss = lossR +lossI;
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                # for p in model.parameters():
                #     p.grad *= -1.0  # Modify gradient for Imag net
                self.optimizer.step()
                TotalEnergy += loss.detach().numpy()
                

                if (j%5 == 0) and (i == len(training_dataloader)-1):
                    pred_R = np.moveaxis(output_R.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    output_dim = int(training_config['trunc_dim'])
                    pred_R = np.array(pred_R)
                    print (pred_R.shape)
                    pred_I = np.moveaxis(output_I.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    # print (pred.shape)
                    pred_I = np.array(pred_I)
                    im = sitk.GetImageFromArray(pred_R[:,:,:,0].reshape(3,output_dim +1,output_dim+1), isVector=False)
                    
                    # sitk.WriteImage(im, './validation_%s.nii'%j, False)

            # ===================log========================            
            print(repr(j) + " Epoch:  " + " Energy:  "+ repr(TotalEnergy));
            

            TotalEnergy=0
            # report_epochs_num = training_config.get('report_per_epochs', 10)      

        print('Training finished')
        return loss
    def pred(self, dataset, scale):
        # Load new data and let go through network
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        # truncdata_shape = (201, 1)
        # predictions = np.zeros(truncdata_shape)
        for dataIdx, data in enumerate(dataloader):
            img_src_R = data['source_R'].to(self.device, dtype = torch.float)
            img_tar_R = data['target_R'].to(self.device, dtype = torch.float)
            img_src_I = data['source_I'].to(self.device, dtype = torch.float)
            img_tar_I = data['target_I'].to(self.device, dtype = torch.float)
            pred_R = self.net(img_src_R,img_tar_R).to(torch.device('cpu')).detach().numpy()
            pred_I = self.net(img_src_I,img_tar_I).to(torch.device('cpu')).detach().numpy()

            # num =self.net(img, img1).to(torch.device('cpu')).detach().numpy()
            # newnum= num.reshape(1,1)
            # predictions[dataIdx,:] = newnum
            # outfile = './src%d.mhd' %dataIdx
            # im = sitk.GetImageFromArray(img.reshape(100,100), isVector=False)
            # sitk.WriteImage(im, outfile, True) 
            # outfile2 = './tar%d.mhd' %dataIdx
            # im1 = sitk.GetImageFromArray(img1.reshape(100,100), isVector=False)
            # sitk.WriteImage(im1, outfile2, True) 
            
            # prediction = np.moveaxis(self.net(img, img1).to(torch.device('cpu')).detach().numpy(),0,-1)            
            # predictions[dataIdx,:] = prediction
            # im = sitk.GetImageFromArray(prediction.reshape(17,17), isVector=False)
            # sitk.WriteImage(im, './pretest.mhd', True) 
        return pred_R, pred_I
    
    def saveLossHistory(self, loss_history, save_filename, report_epochs_num):
        plt.ioff()
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(np.arange(1,(len(loss_history)+1))*report_epochs_num, np.log(loss_history))
        fig.savefig(save_filename, bbox_inches='tight')   # save the figure to file
        plt.close(fig)        
    
    def valid(self, img, save_filename, config):
        # 1. Go through network
        img = torch.from_numpy(img).to(self.device, dtype = torch.float)
        outimg = self.net(img).to(torch.device('cpu')).detach().numpy()
        
        # 2. Take slice as an image
#        if len(np.shape(img)) == 4:
#            # if images are 2D image and img has shape [N,C,H,W]
#            img_sample = outimg[config.get('index', 0), :]            
#        elif len(np.shape(img)) == 5:
#            # if images are 3D image and img has shape [N,C,D,H,W]
#            img_sample_3D = outimg[config.get('index', 0), :]
#            slice_axis = config.get('slice_axis',2)
#            slice_index = config.get('slice_index',0)
#            if slice_index == 'middle': slice_index = int(np.shape(img_sample_3D)[slice_axis]/2)
#            if slice_axis == 0:             
#                img_sample = img_sample_3D[:,slice_index,:,:]
#            elif slice_axis == 1:
#                img_sample = img_sample_3D[:,:,slice_index,:]
#            elif slice_axis == 2:
#                img_sample = img_sample_3D[:,:,:,slice_index]
#        else:
#            raise ValueError(f'Wrong image dimension. \
#                             Should be 4 ([N,C,H,W]) for 2d images \
#                             and 5 ([N,C,D,H,W]) for 3D images, \
#                             but got {len(np.shape(img))}')
        img_sample = slice_img(outimg, config)
        
        # 3. Save slice
        plt.imsave(save_filename, np.squeeze(img_sample), cmap='gray')
    
    
    def save(self, filename_full):
        # Save trained net parameters to file
        # torch.save(self.net.state_dict(), f'../model/{name}.pth')
        # torch.save(self.net.state_dict(), filename_full)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename_full)

    
    def load(self, checkpoint_path):
        # Load saved parameters
        self.continueTraining = True
        checkpoint = torch.load(checkpoint_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.net.parameters())
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.net.eval()
        # self.net.load_state_dict(torch.load(model_path))
        # self.net.eval()
    

def get_loss(config):
    name = config['name']
    para = config['para']
    if name == 'MSE':
        return nn.MSELoss()
    if name == 'TV2D':
        return TVLoss2D(paras.get('TV_weight', 1))
    if name == 'TV3D':
        return TVLoss3D(paras.get('TV_weight', 1))
    if name == 'L1':
        return nn.L1Loss()
    else:
        raise ValueError(f'Unsupported loss type: {name}')

def slice_img(img, config):
    if len(np.shape(img)) == 4:
        # if images are 2D image and img has shape [N,C,H,W]
        img_sample = img[config.get('index', 0), :]            
    elif len(np.shape(img)) == 5:
        # if images are 3D image and img has shape [N,C,D,H,W]
        img_sample_3D = img[config.get('index', 0), :]
#        print(np.shape(img_sample_3D))
        slice_axis = config.get('slice_axis',2)
        slice_index = config.get('slice_index',0)
        if slice_index == 'middle': slice_index = int(np.shape(img_sample_3D)[slice_axis]/2)
        if slice_axis == 0:             
            img_sample = img_sample_3D[:,slice_index,:,:]
        elif slice_axis == 1:
            img_sample = img_sample_3D[:,:,slice_index,:]
        elif slice_axis == 2:
            img_sample = img_sample_3D[:,:,:,slice_index]
    else:
        raise ValueError(f'Wrong image dimension. \
                         Should be 4 ([N,C,H,W]) for 2d images \
                         and 5 ([N,C,D,H,W]) for 3D images, \
                         but got {len(np.shape(img))}')
    return img_sample
        


    

