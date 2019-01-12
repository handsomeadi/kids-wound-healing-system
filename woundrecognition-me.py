# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:32:39 2018

@author: adi
"""

from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten,Activation,Input,GlobalAveragePooling2D,BatchNormalization,Add
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#pre-process
train_path='D:/dataset-wound/train_data'
test_path='D:/dataset-wound/test_data'

traind= ImageDataGenerator(rotation_range=20,width_shift_range=3.0, rescale=1/255.,height_shift_range=3.0).flow_from_directory(train_path, target_size=(224,224) ,classes=['s','b','c','x'], batch_size=10)
testd= ImageDataGenerator(rescale=1/255.).flow_from_directory(test_path,target_size=(224,224) ,classes=['s','b','c','x'], batch_size=10)

img_input = Input(shape=(224,224,3))

x = Conv2D(64,(3,3),activation='relu',padding='same')(img_input)
x = Conv2D(64,(3,3),activation='relu',padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
x = Conv2D(128,(3,3),activation='relu',padding='same')(x)
x = BatchNormalization()(x)     
x = MaxPooling2D((2,2), strides=(2,2))(x)

x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
x = Conv2D(256,(3,3),activation='relu',padding='same')(x)
x = BatchNormalization()(x)     

x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4,activation='softmax')(x)
model = Model(img_input,x)


adam = Adam(lr=1e-5)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint('wound0112me.h5', monitor='val_loss', save_best_only=True)
model.summary()
model.fit_generator(traind, steps_per_epoch=8, validation_data=testd, validation_steps=13, epochs=200, verbose=1, callbacks=[checkpoint])