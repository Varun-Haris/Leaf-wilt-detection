# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:21:00 2019

@author: varun
"""
import csv
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from keras import optimizers, losses
from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, Flatten
from keras.models import Sequential

path = r"E:\NCSU\Fall-19\Neural Networks\projectC2"

class kMeanCluster():
    def __init__(self):
        self.model = Sequential()
        self.image_list = []
        self.image_name_list = []
        
    def datasetPreparation(self):
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
                
        #self.train_x, _, self.test_x, _ = train_test_split(self.train_x, self.train_x, test_size=0.25)
        
    def buildModel(self):
        self.datasetPreparation()
        
        self.model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(128,96,3)))
        self.model.add(MaxPooling2D((2,2), padding='same'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(256, kernel_size=(3,3), activation='relu', input_shape=(128,96,3)))
        self.model.add(MaxPooling2D((2,2), padding='same'))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv2D(512, kernel_size=(3,3), activation='relu', input_shape=(128,96,3)))
        self.model.add(MaxPooling2D((2,2), padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        
        self.model.add(Dense(1024, activation='sigmoid')) # Extracting 1024 features
        
        self.model.summary()
        self.model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam())
        
        '''self.model.fit(self.train_x[:,:,:,0:1], self.train_x[:,:,:,0:1],
                       epochs=10,
                       batch_size=128,
                       verbose=1)'''
        
        self.predictions = self.model.predict(self.train_x, verbose=1)
        #print("All the image features of size 1024 each : {}".format(self.predictions.shape))
        self.predictions = self.predictions.astype('float32').T
        
    def getKMeanClusters(self):        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness,labels,centers = cv2.kmeans(self.predictions,5,None,criteria,10,flags)
        
        A = self.predictions[labels.ravel()==0]
        B = self.predictions[labels.ravel()==1]
        C = self.predictions[labels.ravel()==2]
        D = self.predictions[labels.ravel()==3]
        E = self.predictions[labels.ravel()==4]
        
        ## Visualize the clusters
        plt.scatter(A[:,0],A[:,1],c = 'black')
        plt.scatter(B[:,0],B[:,1],c = 'red')
        plt.scatter(C[:,0],C[:,1],c = 'blue')
        plt.scatter(D[:,0],D[:,1],c = 'green')
        plt.scatter(E[:,0],E[:,1],c = 'orange')
        
        plt.scatter(centers[:,0],centers[:,1],s = 10,c = 'yellow', marker = 's')
        
                
obj = kMeanCluster()
obj.buildModel()
obj.getKMeanClusters()