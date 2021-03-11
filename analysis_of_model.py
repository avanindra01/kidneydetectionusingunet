# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 19:40:37 2021

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
from skimage.morphology import binary_dilation, binary_erosion 

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


model = color_unet()
model.load_weights('./unet_membrane-10-.hdf5')
test_dir = r"C:\Users\Conor\Documents\kidneyproject\train\cut\test"
mask_test_dir = r"C:\Users\Conor\Documents\kidneyproject\train\mask\test"
inner_file_list = os.listdir(test_dir)

acc_quick_t = {10:[],20:[],30:[],40:[],50:[],60:[],70:[],80:[],90:[]}
acc_quick = []
testGene = testGenerator(test_dir,num_image=len(inner_file_list))
for i in range(200):
    cap = next(testGene)
    results = model.predict(cap)
    mask = io.imread(os.path.join(mask_test_dir,inner_file_list[i]), as_gray=True)
    for i in range(1,10,1):
        results_thresh = (results[0,:,:,0]>(i/10))
        '''
        for nn in range(3):
            results_thresh = binary_erosion(results_thresh)
        '''
        acc_quick_t[i*10].append(2*np.sum((mask>0.5)*(results_thresh))/(np.sum(mask>0.5)+np.sum(results_thresh)))
    out_mask = np.zeros([np.shape(mask)[0],np.shape(mask)[1],3])
    #plt.imshow(cap[0,:,:])
    #plt.figure()
    #plt.imshow(results[0,:,:,0]>0.5)
    #plt.figure()
    #plt.imshow(mask)
    #plt.figure()
    #out_mask[:,:,0] = mask
    #out_mask[:,:,1] = results[0,:,:,0]>0.5
    #cv2.imwrite(os.path.join(r'C:\Users\Conor\Documents\kidneyproject\train\outcomes',str(i)+'.png'),out_mask)
    
#print(acc_quick)
#print(np.mean(acc_quick))
for keys in acc_quick_t:
    print(np.mean(acc_quick_t[keys]))
