

import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, \
Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.metrics import classification_report, confusion_matrix


mainDIR = os.listdir('./chest_xray')
print("DATA-load : ", mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'


# train 
os.listdir(train_folder)
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'


print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title : ', norm_pic)
norm_pic_address = train_n + norm_pic


rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic
print('pneumonia picture title : ', sic_pic)


norm_load =Image.open(norm_pic_address)
sic_load =Image.open(sic_address)


f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')
a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.savefig('./src_img/ANN_PNEUMONIA_CNN_00_sample.png', dpi=300, bbox_inches='tight')
plt.close()


model_in = Input(shape = (64, 64, 3))
model = Conv2D(filters=64, kernel_size=(3, 3), activation='relu') (model_in)
model = MaxPooling2D((2, 2)) (model)
model = Conv2D(filters=64, kernel_size=(3, 3), activation='relu') (model)
model = MaxPooling2D((2, 2)) (model)
model = Conv2D(filters=64, kernel_size=(3, 3), activation='relu') (model)
model = MaxPooling2D((2, 2)) (model)
model = Flatten() (model)
model = Dense(units=128, activation='relu') (model)
model = Dense(units=1, activation='sigmoid') (model)
model_fin = Model(inputs = model_in, outputs = model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_folder, target_size = (64, 64),
                                                  batch_size = 32, class_mode = 'binary')


validation_generator = test_datagen.flow_from_directory(val_folder, target_size = (64, 64),
                                                        batch_size = 32, class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_folder, target_size = (64, 64),
                                            batch_size = 32, class_mode = 'binary')
model_fin.summary()


cnn_model = model_fin.fit(training_set, steps_per_epoch = 163, epochs = 10,
                          validation_data = validation_generator, validation_steps = 624)

test_accu = model_fin.evaluate(test_set, steps = 624)

model_fin.save('./models/PNEUMONIA_ANN.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')

Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis = 1)
max(y_pred)


plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show(block=False)
plt.savefig('./src_img/ANN_PNEUMONIA_CNN_01_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()


plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show(block=False)
plt.savefig('./src_img/ANN_PNEUMONIA_CNN_02_LOSS.png', dpi=300, bbox_inches='tight')
plt.close()


