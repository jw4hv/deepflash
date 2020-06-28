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
#               loss = self.criterion(output, img[:,0:1,:,:,:])
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
                # print(output_R.shape)
                #loss2 = self.criterion(output2, label)
                loss = lossR + lossI;
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print("LossInBacth"+ repr(i) + "iteration" + repr(loss.detach().numpy()))
                TotalEnergy += loss.detach().numpy()
                

                if (j%5 == 0) and (i == len(training_dataloader)-1):
                    pred_R = np.moveaxis(output_R.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    
                    pred_R = np.array(pred_R)
                    print (pred_R.shape)
                    pred_I = np.moveaxis(output_I.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    # print (pred.shape)
                    pred_I = np.array(pred_I)
                    im = sitk.GetImageFromArray(pred_R[:,:,:,0].reshape(3,17,17), isVector=False)
                    sitk.WriteImage(im, './prediction_%s.nii'%j, False)

            # ===================log========================            
            print(repr(j) + " Epoch:  " + " Energy:  "+ repr(TotalEnergy));
            

            TotalEnergy=0
            # report_epochs_num = training_config.get('report_per_epochs', 10)            
        print('Training finished')
        return loss
    def pred(self, dataset, scale):
        # Load new data and let go through network
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        truncdata_shape = (201, 1)
        predictions = np.zeros(truncdata_shape)
        for dataIdx, data in enumerate(dataloader):
            img = data['source'].to(self.device, dtype = torch.float)
            img1 = data['target'].to(self.device, dtype = torch.float)

            #print ((self.net(img, img1).to(torch.device('cpu')).detach().numpy())*scale)
            num =self.net(img, img1).to(torch.device('cpu')).detach().numpy()
            newnum= num.reshape(1,1)
            predictions[dataIdx,:] = newnum
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
        return predictions
    
    def saveLossHistory(self, loss_history, save_filename, report_epochs_num):
        # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib        
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
    
    def test(self, dataset):
        # If classification/regression, do pred() and report error
        pass
    
    def save(self, filename_full):
        # Save trained net parameters to file
        # torch.save(self.net.state_dict(), f'../model/{name}.pth')
        # torch.save(self.net.state_dict(), filename_full)
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename_full)

    
    def load(self, checkpoint_path):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
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
    # https://blog.csdn.net/gwplovekimi/article/details/85337689
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
        
import torch.nn as nn
class TVLoss2D(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss2D,self).__init__()
        self.TVLoss_weight = TVLoss_weight
 
    def forward(self, output, truth):
        # [N, C, H, W]
        # [N, C, T, H, W]
        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        count_h = self._tensor_size(output[:,:,1:,:])
        count_w = self._tensor_size(output[:,:,:,1:])
        h_tv = torch.pow((output[:,:,1:,:]-output[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((output[:,:,:,1:]-output[:,:,:,:w_x-1]),2).sum()
        
        l2_loss = torch.pow(output - truth, 2).sum() / (self._tensor_size(output)*batch_size)
        tv_loss = (h_tv/count_h+w_tv/count_w)/batch_size
        return l2_loss + self.TVLoss_weight * tv_loss
#        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
# import torch.nn as nn
class TVLoss3D(nn.Module):
    def __init__(self,TV_weight=1):
        super(TVLoss3D,self).__init__()
        self.TV_weight = TV_weight
 
    def forward(self, output, truth):
        # [N, C, H, W]
        # [N, C, T, H, W]
        N, C, D, H, W = output.shape
        count_d = self._tensor_size(output[:,:,1:,:,:])
        count_h = self._tensor_size(output[:,:,:,1:,:])
        count_w = self._tensor_size(output[:,:,:,:,1:])
        
        d_tv = torch.pow((output[:,1:,:,:]-output[:,:,:D-1,:,:]),2).sum()
        h_tv = torch.pow((output[:,:,1:,:]-output[:,:,:,:H-1,:]),2).sum()
        w_tv = torch.pow((output[:,:,:,1:]-output[:,:,:,:,:W-1]),2).sum()
        
        l2_loss = torch.pow(output - truth, 2).sum() / (self._tensor_size(output)*N)
        tv_loss = (d_tv/count_d + h_tv/count_h + w_tv/count_w)/N
        return l2_loss + self.TV_weight * tv_loss
#        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
 
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
