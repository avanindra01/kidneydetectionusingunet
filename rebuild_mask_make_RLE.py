# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:47:33 2021

@author: Conor
"""

import pandas as pd
import tifffile
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob
import skimage.io as io

import csv


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

input_images = r'C:\Users\Conor\Documents\kidneyproject\test'
#data_file = r'C:\Users\Conor\Documents\kidneyproject\train\cut'
input_masks = r'C:\Users\Conor\Documents\kidneyproject\test\masks'
os.chdir(input_images)

img_list = glob.glob('*.tiff')
#img_list = [img_list[0]]
aspect_ratio = 256
out=[]

out.append('id,encoding\n')
for img in img_list:
    im_id = img[:-5]
    print(im_id)
    image = tifffile.imread(os.path.join(input_images,img))

    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    tru_img_shape=np.shape(image)
    image=image[::2,::2,:]
    img_shape = np.shape(image)
    new_mask = np.zeros([img_shape[0],img_shape[1]],np.uint8)
    small_masks = os.listdir(input_masks)
    how_many = len(small_masks)
    for snn,small_mask in enumerate(small_masks):
        if snn%1000==0:
            print(snn/how_many*100)
        if not(im_id in small_mask):
            continue

        #print('fucking hell')
        coord = small_mask.split('_')
        y = int(coord[0])
        x = int(coord[1])
        #print(small_mask)
        #print(x)
        #print(y)
        x_start = x*aspect_ratio
        x_stop = (x+1)*aspect_ratio
        y_start = y*aspect_ratio
        y_stop = (y+1)*aspect_ratio
        #print(x_start)
        #print(y_start)
        small_mask_load = io.imread(os.path.join(input_masks,small_mask),as_gray=True)
        new_mask[y_start:y_stop,x_start:x_stop] = small_mask_load.astype(np.uint8)

    new_mask = cv2.resize(new_mask,(tru_img_shape[1],tru_img_shape[0]))   
    new_mask =new_mask>1
    rle = mask2rle(new_mask)
    out.append(im_id+','+rle+'\n')
    print(im_id)
csvfile = open('final_output.csv', mode='w',newline='')
csvfile.writelines(out)
csvfile.close()

    



#load image

# calculate image size/256 make zeros array size of iamge


# pastE masks in image by range

#mask2RLE

#write RLE in files

