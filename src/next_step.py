"""Script to forecast next encoded form"""
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
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
predictor = load_model('predictor.h5')


def fun():
    test = []
    dataset = read_csv('output.csv', header=0, index_col=0)
    values = dataset.values
    print(values[1,:].shape)
    preprocessed = values[1,:].reshape((1,1, 1124))
    output= predictor.predict(preprocessed)
    print(output)
    with open("forecast.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


fun()
