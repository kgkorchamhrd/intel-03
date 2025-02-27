from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras import datasets, layers, models, model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
import tensorflow as tf
import numpy as np
import numpy as np  # forlinear algebra
import matplotlib.pyplot as plt  # for plotting things
import os
from PIL import Image  # for reading images

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'
# train
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
# Normal pic
print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)
norm_pic_address = train_n+norm_pic
# Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)

# Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)
# Let's plt these images
f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()
# let's build the CNN model

# cnn = Sequential()
# Convolution
model_in = Input(shape=(64, 64, 3))
model = Flatten()(model_in)
# Fully Connected Layers
model = Dense(activation='relu', units=128)(model)
model = Dense(activation='sigmoid', units=1)(model)
# Compile the Neural network
model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer='adam', loss='binary_crossentropy', metrics=# model.save('transfer_learning_pneumonia.keras')
# with open('history_pneumonia', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)
