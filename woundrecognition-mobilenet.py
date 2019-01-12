# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:32:39 2018
20190112
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

mobile = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalAveragePooling2D()(mobile.output)
x = Dense(1024,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4,activation='softmax')(x)

model = Model(mobile.input, x)
model.summary()

adam = Adam(lr=1e-5)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
checkpoint = ModelCheckpoint('wound0112mobile.h5', monitor='val_loss', save_best_only=True)
model.summary()
model.fit_generator(traind, steps_per_epoch=8, validation_data=testd, validation_steps=13, epochs=200, verbose=1, callbacks=[checkpoint])
