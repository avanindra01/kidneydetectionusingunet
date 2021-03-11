# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:22:29 2021

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
from skimage import exposure


# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def rle2mask(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = 254
    return img.reshape(shape[0:2])

directory = r'C:\Users\Conor\Documents\kidneyproject\train'
train_dir = r'C:\Users\Conor\Documents\kidneyproject\train\cut'
mask_dir = r'C:\Users\Conor\Documents\kidneyproject\train\mask'
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
    os.makedirs(os.path.join(train_dir,'test'))
    os.makedirs(os.path.join(train_dir,'train'))
    os.makedirs(mask_dir)
    os.makedirs(os.path.join(mask_dir,'test'))
    os.makedirs(os.path.join(mask_dir,'train'))



image_data = pd.read_csv(os.path.join(directory,'a.csv'))
test_train_split = 0.75
#im_name = '2f6ecfcdf.tiff'
#im_id = im_name[:-5]
#image = tifffile.imread(os.path.join(directory,'2f6ecfcdf.tiff'))
#mask = rle2mask(image_data[image_data['id']==im_id]["encoding"].values[0],np.shape(image))
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
    mask = rle2mask(image_data[image_data['id']==im_id]["encoding"].values[0],[img_shape[1],img_shape[0]])
    mask=mask.T
    #downsampling images to ensure more of a fetuere is in sub image....
    image = image[::2,::2,:]
    mask = mask[::2,::2]
    #GIVEN no cells on the edge, we are gonna be lazy here
    img_shape = np.shape(image)
    y_imgs = int(np.floor(img_shape[0]/aspect_ratio))
    x_imgs = int(np.floor(img_shape[1]/aspect_ratio))
    
    for x in range(x_imgs-1):
        for y in range(y_imgs-1):
            if np.random.random() >test_train_split:
                dir_hinge='test' 
            else:
                dir_hinge = 'train'
            temp_img = image[y*aspect_ratio:(y+1)*aspect_ratio,x*aspect_ratio:(x+1)*aspect_ratio,:]
            if np.sum(temp_img)==0:
                continue
            temp_mask = mask[y*aspect_ratio:(y+1)*aspect_ratio,x*aspect_ratio:(x+1)*aspect_ratio]
            if np.sum(temp_mask)==0:
                continue #this helps balance the groups a little by ecluding some of the images which have no mask
            #cv2.imwrite(os.path.join(train_dir,dir_hinge,str(y)+'_'+str(x)+'_'+im_id+'.png'),temp_img)
            #cv2.imwrite(os.path.join(mask_dir,dir_hinge,str(y)+'_'+str(x)+'_'+im_id+'.png'),temp_mask)
    print(im_id)
    
    