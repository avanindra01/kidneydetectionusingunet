# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:14:16 2021

@author: Conor
"""

from tensorflow.keras import Model, Input 
from keras.layers import Conv2D, MaxPooling2D, Dropout,UpSampling2D,concatenate
from  keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import cv2

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
'''
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
'''

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        img = img-np.mean(img)
        img = img/np.std(img)
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)    




def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode ="rgb",
                   mask_color_mode = "grayscale", image_save_prefix ="image", mask_save_prefix ="mask",
                   flag_multi_class = False, num_class = 2, save_to_dir = None, target_size = (256,256), seed =1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator,mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    file_list = os.listdir(test_path)
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,file_list[i]), as_gray=False)
        img= img/255
        img = img-np.mean(img)
        img = img/np.std(img)
        img = trans.resize(img,target_size)
        #img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img
        
def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255
def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return numerator / (denominator + tf.keras.backend.epsilon())

def loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon())   

'''
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        '''

def color_unet(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = Conv2D(128,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
    
    
    
    conv5 = Conv2D(1024,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv6)
    
    
    up7 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv7)
    
    up8 = Conv2D(128,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv8)
    
    
    up9 = Conv2D(128,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(256,(3,3),activation = 'relu', padding = 'same', kernel_initializer='he_normal')(conv9)
                                                                                                 
    conv10 = Conv2D(1,1,activation='sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = 'sgd', loss = loss, metrics = [dice_coefficient])
    
    model.summary()
    return(model)

def unet(pretrained_weights = None,input_size = (256,256,1)):    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = 'sgd', loss = loss, metrics = [dice_coefficient])
    
    model.summary()
    return(model)
 
myGene = trainGenerator(2,r'C:\Users\Conor\Documents\kidneyproject\train','cut','mask',data_gen_args,save_to_dir = None)

model_checkpoint = ModelCheckpoint('unet_membrane-{epoch:02d}-.hdf5', monitor='loss',verbose=1, save_best_only=False)
model = color_unet()
history = model.fit_generator(myGene,steps_per_epoch=2000,epochs=10,callbacks=[model_checkpoint])


    


