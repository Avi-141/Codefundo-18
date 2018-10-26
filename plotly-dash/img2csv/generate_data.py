# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import sys
from PIL import Image
# create the data folder if it doesn't exist yet
if not os.path.exists("data"):
    os.makedirs("data")
vals = []
index = []
count = 0
def ProcessImage(file):
    img = Image.open(file)
    width, height = img.size
    img = img.resize((100, 100), Image.ANTIALIAS)
    img_grey = img.convert('L')
    value = np.asarray(img_grey.getdata(), dtype=np.int)
    value = value.flatten()
    vals.append(value)

path1 = './putimageshere/cyclone1'
path2 = './putimageshere/cyclone2'
path3 = './putimageshere/cyclone3'

files = next(os.walk(path1))[2]
for file in files:
    if(re.search(".jpeg|.JPEG",file)):
        ProcessImage(path1+"/"+file)
        index.append(0)

print("Loaded cyclone 1")

files = next(os.walk(path2))[2]
for file in files:
    if(re.search(".jpeg|.JPEG",file)):
        ProcessImage(path2+"/"+file)
        index.append(1)


print("Loaded cyclone 2")

files = next(os.walk(path3))[2]
for file in files:
    if(re.search(".jpeg|.JPEG",file)):
        ProcessImage(path3+"/"+file)
        index.append(2)


print("Loaded cyclone 3")

print("Dataset loaded.")

# Flatten the array, and normalize it
# We will select the integer values to be the index
df = pd.DataFrame(vals,index=index)

sample_size = 1
samples = df.sample(n=index.__len__(), random_state=1234)
dataset_name = "test"

samples.to_csv(f"data/{dataset_name}_{sample_size}_input.csv", index=False)
pd.DataFrame(samples.index).to_csv(f"data/{dataset_name}_{sample_size}_labels.csv", index=False)

print("CSV files created.")
