# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:06:56 2021

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
import cv2

directory = r'C:\Users\Conor\Documents\kidneyproject\test'
os.chdir(directory)
img_list = glob.glob('*.tiff')
#img_list = [img_list[0]]
aspect_ratio = 256

for img in img_list:
    im_id = img[:-5]
    print(im_id)
    image = tifffile.imread(os.path.join(directory,img))
    print(image.shape)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    img_shape = np.shape(image)
    print(img_shape)
    image = image[::2,::2,:]
    #GIVEN no cells on the edge, we are gonna be lazy here
    img_shape = np.shape(image)
    y_imgs = int(np.floor(img_shape[0]/aspect_ratio))
    x_imgs = int(np.floor(img_shape[1]/aspect_ratio))
    
    for x in range(x_imgs-1):
        for y in range(y_imgs-1):
            temp_img = image[y*aspect_ratio:(y+1)*aspect_ratio,x*aspect_ratio:(x+1)*aspect_ratio,:]
            cv2.imwrite(os.path.join(directory,'small',str(y)+'_'+str(x)+'_'+im_id+'.png'),temp_img)
    print(im_id)
    
    