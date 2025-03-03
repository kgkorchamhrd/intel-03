import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img


print("Current Working Directory:", os.getcwd())

minDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_flder = './chest_xray/val/'
test_folder = './chest_xray/test/'

os.lsitdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
print(len(os.listdir(train_n)))

rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = ost.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic

print('pneumonia picture title:', sic_pic)

