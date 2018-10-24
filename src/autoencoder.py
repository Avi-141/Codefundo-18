#import all
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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import load_model
from keras import regularizers
from keras.layers.normalization import BatchNormalization



"""func to plot loss on the graph"""

def plot_train_history_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

"""To disply reconstructed images """
def display_reconstructed(x_test, decoded_imgs, n=10):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(300, 300))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


"""Func to convert images to gray"""
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


"""Import the datset from dir and create x_train"""
x_train = [] #creae an empty list to hold the training data

mypath='path_to_training_data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):

    x_train.append(rgb2gray(cv2.resize(cv2.imread( join(mypath,onlyfiles[n]) ), (300,300))))

x_test = []
mypath='path_to_test_data'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    x_test.append(rgb2gray(cv2.resize(cv2.imread( join(mypath,onlyfiles[n]) ), (300,300))))

x_train = np.array(x_train)
x_test = np.array(x_test)                        #turn images into numpy arrays and normalize them

x_train = x_train.astype('float32') / 255.
print(x_train.shape)


x_test = x_test.astype('float32') / 255.
print(x_test.shape)


print('x_train.shape:', x_train.shape)

#setting up
mb_size = 64
z_dim = 100
X_dim = x_train.shape[1]
print(X_dim)


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape, x_test.shape)

input_dim = x_train.shape[1]
encoding_dim = 32

compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

x_train = x_train.reshape((len(x_train), 300, 300, 1))
x_test = x_test.reshape((len(x_test), 300, 300, 1))

autoencoder = Sequential()

# Encoder Layers
# Encoder Layers
encoder = Sequential()
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(300,300,1)))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((5, 5), padding='same'))
encoder.add(Conv2D(5, (3, 3), activation='relu', padding='same'))

encoder.add(Flatten())


decoder = Sequential()
decoder.add(Dense(units = 1125))
decoder.add(Reshape((15, 15, 5)))
decoder.add(UpSampling2D((5, 5)))
decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

autoencoder = Sequential()
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.summary()

#encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('flatten_1').output)
#encoder.summary()
#decoder = Model(inputs= autoencoder.get_layer('flatten_1').output, outputs =autoencoder.get_layer('conv2d_8').output )
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=8,
                validation_data=(x_test, x_test))
num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

#save the models for furthur use
autoencoder.save('autoencoder_v1.h5')
encoder.save('encoder_v1.h5')
decoder.save('decoder_v1.h5')

plt.figure(figsize=(18, 4))


plt.figure(figsize=(18, 4))

for i, image_idx in enumerate(random_test_images):
    # plot original image
    ax = plt.subplot(3, num_images, i + 1)
    plt.imshow(x_test[image_idx].reshape(300, 300))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot encoded image
    ax = plt.subplot(3, num_images, num_images + i + 1)
    plt.imshow(encoded_imgs[image_idx].reshape(75, 15))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # plot reconstructed image
    ax = plt.subplot(3, num_images, 2*num_images + i + 1)
    plt.imshow(decoded_imgs[image_idx].reshape(300, 300))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
