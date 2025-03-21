# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 15:17:53 2023

@author: bellr
"""

import argparse
import os
import multiprocessing
import pandas as pd


from move_files_for_inference_v2 import move_files_for_inference
from Disk_Load_Weights_And_Infer import infer
from organize_inference_for_Majority_vote_Disk_2 import organize_for_majority_vote 
from majority_vote_Disk_v3_Multi import vote




def split_folders(folder_list, num_splits):
    avg_split = len(folder_list) // num_splits
    remaining = len(folder_list) % num_splits

    split_folders = []
    start = 0
    for i in range(num_splits):
        end = start + avg_split + (1 if i < remaining else 0)
        split_folders.append(folder_list[start:end])
        start = end

    return split_folders


    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='Run_Disk_Infer',
                        description='This program infers Disk tissue segmentation based on Bell et al models'
                        )
    parser.add_argument('QuPath_Folder_Path', nargs = '?',
                        help='Please set the path (String) to the QuPath Project Folder', type = str)
    parser.add_argument('Model_Path', nargs = '?',
                        help='Please set the full path (String) to the Model', type = str)
    parser.add_argument('Delete_Temp_Files', nargs = '?',  default = False, type = bool,
                        help='Indicate (True or False) if you want to delete the Temp Files Created (Default = True) which include the folders : tiles, Results_Temp, Results_from_Inference, Org_for_Inference')
    parser.add_argument('Infer_Batch_Size', nargs = '?', default = 10,
                        help='Indicate How Many (Integer) Images to Process at a Time - Dependent on GPU VRAM - Default = 10', type = int)
    parser.add_argument('Tile_Size', nargs = '?', const = 1, default = 512,
                        help='Indicate the tile size (Integer) exported by QuPath - Default = 512', type = int)
    parser.add_argument('Downsample', nargs = '?', const = 1, default = 1,
                        help='Indicate the Downsample rate (Integer) used in the export by QuPath - Default = 4', type = int)
    parser.add_argument('Instances', nargs = '?', type=int, default=3,
                        help='Number of instances (Integer) to run in parallel')
    
    args = parser.parse_args()
    
    path = args.QuPath_Folder_Path
    model_path = args.Model_Path
    delete_temp_files = args.Delete_Temp_Files
    infer_batch_size = args.Infer_Batch_Size
    tile_size = args.Tile_Size
    downsample = args.Downsample
    orig_size = downsample * tile_size
    skip_steps = [0]
    
    n_classes = 7
    args = parser.parse_args()
    instances = args.Instances
    

    
    if 1 in skip_steps:
        print('Skipping 1/4: Move Files For Inference')
        print('')
    else:
        print('Running 1/4: Move Files For Inference')
        move_files_for_inference(path)
        print('')

    if 2 in skip_steps:
        print('Skipping 2/4: Inference')
        print('')
    else:
        print('Running 2/4: Inference')
        infer(path, infer_batch_size, model_path)
        print('')
    
    if 3 in skip_steps:
        print('Skipping 3/4: Organize for Majority Vote')
    else:
        print('Running 3/4: Organize for Majority Vote')
        if not os.path.isdir(f'{path}\\Majority_Vote_Results'):
            os.mkdir(f'{path}\\Majority_Vote_Results')
        organize_for_majority_vote(path)
        print('')
    
    
    if 4 in skip_steps:
        print('Skipping 4/4: Majority Vote')
    
    
    else:
        slide_folders = os.listdir(f'{path}\\Results_Temp\\')
        
        if "Thumbs.db" in slide_folders:
            slide_folders.remove("Thumbs.db")
        
    
        print('Running 4/4: Majority Vote')

    
        folder_splits = split_folders(slide_folders, instances)
    
        processes = []
        for split in folder_splits:
            process = multiprocessing.Process(target=vote, args=(path, split, n_classes, orig_size, tile_size))
            processes.append(process)
            process.start()
    
        for process in processes:
            process.join()
    

    # if delete_temp_files == True:
        
    #     import shutil
        
    #     shutil.rmtree(path + 'tiles')
    #     shutil.rmtree(path + 'Results_Temp')
    #     shutil.rmtree(path + 'Results_from_Inference')
    #     shutil.rmtree(path + 'Org_for_Inference')
