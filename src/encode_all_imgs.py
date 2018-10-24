import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import load_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization

"""Load model"""
encoder = load_model('denoise_encoder.h5')
encoder.summary()

"""Func to convert images to gray"""
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

"""Import the datset from dir and create x_train"""
x_train = []

mypath='path_to_training_data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):

    x_train.append(rgb2gray(cv2.resize(cv2.imread( join(mypath,onlyfiles[n]) ), (300,300))))

x_train = np.array(x_train)
x_train = x_train.astype('float32') / 255.
print('x_train.shape:', x_train.shape)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train = x_train.reshape((len(x_train), 300, 300, 1))
print('x_train.shape:', x_train.shape)

"""Use the encoder to convert all images into encoded form"""
encoded_imgs = []
encoded_imgs = encoder.predict(x_train)

print(encoded_imgs.shape)

"""Write the encoded image matrix to a csv file"""
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(encoded_imgs)
