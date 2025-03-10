'''https://github.com/usuyama/pytorch-unet'''

#%% Imports

import copy
import time
from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
import cv2
import random
from tqdm import tqdm 
from torch import nn
import transforms_inference as T
#import utils
from imgaug import augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapOnImage
import torchvision
import skimage.io as io
from datetime import datetime
from collections import defaultdict
import torch.nn.functional as F
import csv
import tifffile
#import pandas as pd


def infer(path, batch_size, model_path):
    
    
    #%%## Set Hyperparamerters and Paths
    
    
    saved_weight_path = model_path 

    data_path = path
    
    root_path = path
    
    
    tiles = '//Org_for_Inference'
    folder_name = '//Results_from_Inference'
    
    if not os.path.isdir(f'{root_path}\\Results_from_Inference'):
        os.mkdir(f'{root_path}\\Results_from_Inference')
    
    
    
    root_path_T = data_path + tiles
    
    batch_size = batch_size

    n_classes = 7
    achitecture = 'efficientnet-b5'

    
    
    ## Make sure the model is the same as the loaded model
    # 1 = unet, 2 = unet++, 3 = PSPnet, 4 = deeplavV3
    model_id =  2
    
    
    ach = achitecture[-2:]
    
    if model_id == 1:
        model_name = 'UNET'
    if model_id == 2:
        model_name = 'UNET++'
    if model_id == 3:
        model_name = 'PSPnet'
    if model_id == 4:
        model_name = 'DeepLabV3'
    
    
    
    
    #%%
    
    
    def seed_everything(seed=1234):                                                  
        random.seed(seed)                                                            
        torch.manual_seed(seed)                                                      
        torch.cuda.manual_seed_all(seed)                                             
        np.random.seed(seed)                                                         
        os.environ['PYTHONHASHSEED'] = str(seed)                                     
        torch.backends.cudnn.deterministic = True                                    
        torch.backends.cudnn.benchmark = False 
    
    seed_everything(1234) 
    
    # Contruct DataLoader
    class KneeDataset(object):
        def __init__(self, root, imaug, transforms):
            self.root = root
            #self.transforms = transforms
            # load all image files, sorting them to
            # ensure that they are aligned
            self.path = os.listdir(root)
            self.masks = []
            self.origs = []
            self.subdirs = []
            self.filenames = []
            self.transforms = transforms
            self.imaug = imaug
            # no_duplicates = os.listdir('R:\\MK Anti-TNF Study\\Results_From_Inference\\')
            # no_duplicates = [x[:-4] for x in no_duplicates]
            # no_duplicates = [x + '.jpg' for x in no_duplicates]
    
            #Unique to the way QuPath exports the tiles with the names
            self.path_img = [ x for x in self.path if ".tif" not in x ]
            # self.path_img = [ x for x in self.path_img if x not in no_duplicates] 
            
    
        def __getitem__(self, idx):
    
            files = self.path_img[idx]
    
            img = cv2.imread(self.root+files)
    
    
            img = Image.fromarray(img, mode = 'RGB')
            
            
            if self.transforms is not None:
                img  = self.transforms(img)
    
    
            return img.float(), files
            
    
        def __len__(self):
            return int(len(self.path_img)) # Cut len in half bc we have the masks in the same path
    
    #%%
    
    def get_transform(train):
        transforms = []
        if train:
            transforms.append(T.ToTensor())
            transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))
        else:
            transforms.append(T.ToTensor())
            transforms.append(T.Normalization([.858, .670, .823], [.098, .164, .050]))  
        return T.Compose(transforms)
    
    #%%
    
    dataset_inference = KneeDataset(f'{root_path_T}\\', None, get_transform(train=False))
    
    # dataloaders = {
    
    #     'inference': DataLoader(dataset_inference, batch_size=batch_size, shuffle=True, num_workers=0)
    # }
    
    inference_loader = DataLoader(dataset_inference, batch_size=batch_size, shuffle=True, num_workers=0)
    #%%
    
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    
    import segmentation_models_pytorch as smp
    
    
    if model_id == 1:
        model = smp.Unet(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 2:
        model = smp.UnetPlusPlus(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 3:
        #model = smp.PSPNet(achitecture, encoder_weights='imagenet', classes = n_classes, in_channels=3, activation = None)
    
        model = smp.PSPNet(achitecture, classes = n_classes, in_channels=3, activation = None)
    
    if model_id == 4:
        model = smp.DeepLabV3Plus(achitecture, encoder_weights='imagenet', classes= n_classes, in_channels=3, activation = None)
    
    # check to make sure of model compatibility

    
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(saved_weight_path))
    model.eval()
    
    
    model = model.to(device)
    
            
    #%%
    from os.path import exists
    
    with torch.no_grad():
              
        for data_fin in tqdm(inference_loader):
            inputs, name = data_fin[0].to(device), data_fin[1] #,  data_fin[2]
          
            outputs = model(inputs)   
            #outputs = F.sigmoid(outputs)
            outputs = torch.sigmoid(outputs)
            #print(outputs.shape)   
        
            for j in range(outputs.shape[0]):
    
                current_name = name[j]
                current_name = current_name[:-4]        
                # if len(os.listdir(f'{root_path}\\{ach}_{model_name}_Eval_Test\\')) > 5:
                #      break
                
            
            
                if outputs[j].shape[0] > 0:
                    # if not os.path.isdir(f'{root_path}\\{folder_name}\\{current_name}\\'):
                    #     os.mkdir(f'{root_path}\\{folder_name}\\{current_name}\\')
        
                    
        
                    temp = outputs[j,:,:,:]*255             
                    temp = temp.cpu().numpy().astype(np.uint8)
                    if exists(f'{root_path}\\{folder_name}\\{current_name}.tif'):
                        continue
                    else:
                        tifffile.imwrite(f'{root_path}\\{folder_name}\\{current_name}.tif', temp, photometric='minisblack')
                    
                    
