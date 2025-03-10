# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:31:19 2021

@author: richard + matt
"""

#majority_vote_of Predictions of overlapping tiles

import os
import numpy as np
import skimage.io as io

from tqdm import tqdm
from scipy.stats import mode
import cv2

from skimage.transform import  resize

def findMinDiff(arr, n):
 
    # Sort array in non-decreasing order
    arr = sorted(arr)
 
    # Initialize difference as infinite
    diff = 10**20
 
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(n-1):
        if arr[i+1] - arr[i] < diff:
            diff = arr[i+1] - arr[i]
 
    # Return min diff
    return diff

def vote(path, folders, n_classes, size, im_size):
    
    if n_classes == 7:
        Class_Names = ['NP', 'AF','Granulation','Endplate', 'Bone', 'Ligament', 'Growth plate']
    
    
    save_path = f'{path}\\Majority_Vote_Results\\'
    root_path = f'{path}\\Results_Temp\\'
    
    #print(root_path)
    
    if not os.path.isdir(f'{save_path}'):
        os.mkdir(f'{save_path}')
    
    #folders = os.listdir(root_path)
    print(folders)
    
    for slides in tqdm(folders):
        
        images = os.listdir(f'{root_path}\\{slides}')
        format_name = images[0].split('=')
    
        x = []
        y = []
        #print(len(images))
        for i in images: 
            name_split_1 = i.split('=')
            #print(name_split_1)
            x_ = name_split_1[1].split(',')
            #print(x_[0])
            x.append(int(x_[0]))
            y_ = name_split_1[2].split(',')
            y.append(int(y_[0]))
            #print(y_)
        
    
        x = list(set(x))
        y = list(set(y))
        max_x = max(x)
        max_y = max(y)
    
        min_x = min(x)
        min_y = min(y)

        
        n = len(x)
        spacing = findMinDiff(x, n)

        
        print('start')
        

    
    
        
        for ii in range(min_x, max_x + 1 + spacing, spacing):
            for jj in range(min_y, max_y + 1 + spacing, spacing):
    

                
                range_interest_x = [ww for ww in x if ii - size/3 <= ww <=ii]
                range_interest_y = [ww for ww in y if jj - size/3 <= ww <=jj]
    
                preds= []
                    
                for aa in range_interest_x:
                    for bb in range_interest_y:
    
                                
                        format_name_main = format_name.copy()
    
                        format_name_main[1] = f'{aa},y'
                        format_name_main[2] = f'{bb},w'
    
                        current_image = ('=').join(format_name_main)
    
                        if range_interest_x == []:
                            aa = ii 
             
                        if range_interest_y == []:
                            bb = jj 
    
    
                        dis_ii = (ii-aa)/spacing
                        dis_jj = (jj-bb)/spacing
    
                        if dis_ii == 0:
                            dis_ii = 1
                            
                        if dis_jj == 0:
                            dis_jj = 1
    
                        #print(ii, dis_ii)
    
                        pix = int((spacing*im_size)/size)
    
    
                        try:
                            prediction_current = io.imread(f'{root_path}//{slides}//{current_image}')
                            prediction_current = np.swapaxes(prediction_current, 0, 1)
                            #print(mask.shape)
                            prediction_current = np.swapaxes(prediction_current, 1, 2)
                            
                            prediction_current = prediction_current[int(pix*dis_jj):int(pix*dis_jj)+int(pix), int(pix*dis_ii):int(pix*dis_ii)+int(pix), :]
    
                            if (prediction_current.shape[0]) < pix or (prediction_current.shape[1] < pix):
                                prediction_current = resize(prediction_current, (prediction_current.shape[2],pix,pix),anti_aliasing=True)
    
                            
    
                    
                            prediction_current[prediction_current < 75] = 0
    
    
    
                            prediction_background = np.sum(prediction_current, axis = 2)
    
    
                            prediction_background[prediction_background != 0 ] = 255
                            
    
                            prediction_background = 255 - prediction_background.reshape((pix, pix, 1))
                            #print(prediction_current.min(), prediction_current.max(),prediction_background.min(), prediction_background.max())
    
                            prediction_current = np.append(prediction_current, prediction_background, axis = 2)
    
                            #if ii > 688:
                            #    print(current_image, ii, jj)
                            #    plt.imshow(prediction_current[9,:,:])
                            #    plt.show()
                            #    plt.imshow(prediction_current[7,:,:])
                            #    plt.show()
    
                            pred_thresh = np.argmax(prediction_current, axis = 2)
                            
                            preds.append(pred_thresh)
                        except:
                            pass
    
                if len(preds) >= 1:
                    fin_pred = np.stack(preds,axis = 0)
    
                    if fin_pred.all() != n_classes :
                        
                        done = mode(fin_pred, axis = 0)
                        
        
                        done = done[0].reshape((pix, pix))
                        #print(done.min(), done.max())
        
                        if not os.path.isdir(f'{save_path}\\{slides}'):
                            os.mkdir(f'{save_path}\\{slides}')
                            
                        #for layers in range(len(Class_Names)): 
                            # if not os.path.isdir(f'{save_path}\\{slides}\\{Class_Names[layers]}'):
                            #     os.mkdir(f'{save_path}\\{slides}\\{Class_Names[layers]}')
        
                        for iclass in range(n_classes):
        
                            idx = np.argwhere(done == iclass)
                            #print(idx)
        
                            done_copy = np.zeros((pix,pix)).astype(np.uint8)
        
                            done_copy[idx[:,0], idx[:,1]] = 255
        
                            new_format_name = format_name.copy()
        
                            test_name = new_format_name[0].split('x')[0][:-2]
                            
                            x_real = ii + spacing
                            y_real = jj + spacing
                            
                            new_format_name[0] = f'{test_name} 5x {Class_Names[iclass]} [x'
                            new_format_name[1] = f'{x_real},y'
                            new_format_name[2] = f'{y_real},w'
                            new_format_name[3] = f'{spacing},h'
                            new_format_name[4] = f'{spacing}].tif'
        
                            current_image_fin = ('=').join(new_format_name)
                            done_copy = done_copy.astype(np.uint8)
        
        
    
        
                            if np.sum(done_copy) > 0:
                                cv2.imwrite(f'{save_path}\\{slides}\\{current_image_fin}', done_copy)        
 
                                