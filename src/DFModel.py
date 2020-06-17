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
#                print(img.device)
                #img2 = torch.cat((img, img1), 1)
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

    def trainDeepFlash(self, training_dataset, training_config, testing_dataset, valid_img = None, expPath = None):
        # Create dataloader
        training_dataloader = DataLoader(training_dataset, batch_size=training_config['batch_size'],
                        shuffle=False)

        # print (training_dataset[0]['gtru'].shape)
        testing_dataloader = DataLoader(testing_dataset, batch_size=1,
                        shuffle=False)
 

        # Set Optimizer
        if self.continueTraining == False:        
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=training_config['learning_rate'],
                                weight_decay=1e-6)
        TotalEnergy = 0
        print("Training started") 
        for j, epoch in enumerate(range(1, training_config['epochs_num']  + 1)):
            for i, data in enumerate(training_dataloader):      
                img = data['source'].to(self.device, dtype = torch.float)
                # print (img.shape)
                img1 = data['target'].to(self.device, dtype = torch.float)
                #label = (1, 1, 16, 16, 16)
                label = data['gtru'].to(self.device, dtype = torch.float)

                # outfile = './out%d.mhd'%i
                # im = sitk.GetImageFromArray(img[9], isVector=False)
                # sitk.WriteImage(im, outfile, True) 

                # source = sitk.GetImageFromArray(img, isVector=False)
                # target = sitk.GetImageFromArray(img1, isVector=False)
                # sitk.WriteImage(source, './source_%s.mhd'%(j*(training_config['epochs_num'])+i), True)
                # sitk.WriteImage(target, './target_%s.mhd'%(j*(training_config['epochs_num'])+i), True)

                # print(label.shape)
#                print(img.device)
                #img2 = torch.cat((img, img1), 1)
                # ===================forward=====================
                output = self.net(img,img1)
                #output2 = self.net(img,img1)
#               loss = self.criterion(output, img[:,0:1,:,:,:])
                loss1 = self.criterion(output, label)
                #loss2 = self.criterion(output2, label)
                loss = loss1 ; #+loss2;
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #print("LossInBacth"+ repr(i) + "iteration" + repr(loss.detach().numpy()))
                TotalEnergy += loss.detach().numpy()
                

                if (j%5 == 0) and (i == len(training_dataloader)-1):
                    
                #     # write_file = "output%s.csv"%j
                #     # with open(write_file, "w") as output:
                #     #     for idx, line in enumerate(testing_dataloader):
                #     #         testsrc = line['source'].to(self.device, dtype = torch.float)
                #     #         testtar = line['target'].to(self.device, dtype = torch.float)

                #     #         # print (testtar.shape)
                #     #         # outfile = './out%d.mhd'%idx
                #     #         # im = sitk.GetImageFromArray(testtar, isVector=False)
                #     #         # sitk.WriteImage(im, outfile, True) 

                #     #         testoutput = self.net(testsrc,testtar)
                #     #         testoutput = testoutput.to(torch.device('cpu')).detach().numpy()
                #     #         testsave = testoutput.reshape(1)
                #     #         output.write(str(testsave)+'\n')

                #     # with open('prediction%s.csv'%j,'wb') as file:
                #     #     for line in text:
                #     #         file.write(line)
                #     #         file.write('\n')
                #     # print(output.shape)
                    pred = np.moveaxis(output.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    # print (pred.shape)
                    pred = np.array(pred)
                    print(pred.shape)
                    im = sitk.GetImageFromArray(pred[:,:,:,0].reshape(2,17,17), isVector=False)
                    sitk.WriteImage(im, './prediction_%s.nii'%j, False)

                    # pred2 = np.moveaxis(label.to(torch.device('cpu')).detach().numpy(),0,-1) 
                    # im2 = sitk.GetImageFromArray(pred2[:,:,:,0].reshape(17,17), isVector=False)
                    # sitk.WriteImage(im2, './groundtruth_%s.mhd'%j, True)
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

#def net_sample_test(net, img, save_filename, save_index = None):
#    # 1. Go through network
#    outimg = net(img).to(torch.device('cpu')).detach().numpy()
#    # 2. Take slice as an image
#    
#    # 3. Save slice
#    outimg = net(img).to(torch.device('cpu')).detach().numpy()
#    if len(outimg.shape) == 4:
        


        
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
