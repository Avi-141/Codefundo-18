"""Script to take an image, encode it, use LSTM network to predict t+1 encoded form
reconstruct the image from encoded form, and overlay it on input image"""
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, Dropout, Activation, Flatten, Reshape
from keras.models import load_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

"""Load model"""
encoder = load_model('denoise_encoder.h5')
encoder.summary()
decoder = load_model('decoder_v1.h5')
"""Func to convert images to gray"""
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

"""func to test image"""

def output(image):
    test = []
    test.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    test = np.array(test)
    test = test.astype('float32') / 255.

    test = test.reshape((len(test), np.prod(test.shape[1:])))
    test = test.reshape((len(test), 300, 300, 1))
    encoded = encoder.predict(test)
    output = decoder.predict(encoded)
    print(output.shape)
    background = test.reshape(300, 300)
    overlay = output.reshape(300, 300)
    added_image = cv2.addWeighted(background,0.4,overlay,0.1,0)


    plt.figure(figsize=(25, 25))
    plt.imshow(encoded.reshape(75, 15))
    plt.show()

    return added_image

final = output(image)
plt.imshow(final.reshape(300, 300))
plt.show()
