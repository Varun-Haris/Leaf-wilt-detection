# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:17:26 2019

@author: varun
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split

from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D
from keras.models import Model
#from keras.utils import plot_model

from keras import backend
backend.set_image_data_format('channels_last')
path = r"E:\NCSU\Fall-19\Neural Networks\projectC2"

class vae():
    def __init__(self):
        self.image_name_list = []
        self.image_list = []
    
    def prepareDataset(self):        
        for root,_,files in os.walk(path):
            current_directory = os.path.abspath(root)
            for f in files:
                name, ext = os.path.splitext(f)
                if ext == '.jpg':
                    current_image_path = os.path.join(current_directory, f)
                    self.image_name_list.append(current_image_path[-9:-1]+current_image_path[-1])
                    img = cv2.imread(current_image_path)
                    self.image_list.append(cv2.resize(img, (96,128)))
            
        self.train_x = np.asarray(self.image_list)
        
        #self.train_x.reshape(len(self.train_x),128,96,3)
        self.train_x = self.train_x.astype('float')/255.
        print("Shape of training data: {}".format(self.train_x.shape))
        
        self.train_x, self.test_x = train_test_split(self.train_x, test_size=0.33)
                
    def buildModel(self):
        self.prepareDataset()
        
        ## Encoder model
        input_img = Input(shape=(128,96,1), name='input_1')

        x = Conv2D(2048, (3, 3), activation='relu', padding='same', name='conv_1')(input_img)
        x = MaxPooling2D((2, 2), padding='same', name='max_1')(x)
        
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv_2')(x)
        x = MaxPooling2D((2, 2), padding='same', name='max_2')(x)
        
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv_3')(x)
        encoded = MaxPooling2D((2, 2), padding='same', name='max_3')(x)

        
        ## Decoder Model
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv_4')(encoded)
        x = UpSampling2D((2, 2), name='up_1')(x)
        
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv_5')(x)
        x = UpSampling2D((2, 2), name='up_2')(x)
        
        x = Conv2D(2048, (3, 3), activation='relu', padding='same', name='conv_6')(x)
        x = UpSampling2D((2, 2), name='up_3')(x)
        
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        trained_model = self.autoencoder.fit(self.train_x[:,:,:,0:1], self.train_x[:,:,:,0:1], 
                                                  epochs=10, 
                                                  batch_size=64, 
                                                  shuffle=True, 
                                                  verbose=1,
                                                  validation_data=(self.test_x[:,:,:,0:1], self.test_x[:,:,:,0:1]))
        
        loss = trained_model.history['loss']
        val_loss = trained_model.history['val_loss']
        epochs = range(len(self.loss))
        
        plt.plot(epochs, loss, 'r', label = 'Training loss')
        plt.plot(epochs, val_loss, 'b', label = 'validation loss')
        plt.legend()
        plt.show()
        
    def getPredictions(self):
        self.decoded_imgs = self.autoencoder.predict(self.test_x)

obj = vae()
obj.buildModel()