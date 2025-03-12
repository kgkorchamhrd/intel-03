import numpy as np
import matplotlib.pyplot as plt 
import os
from PIL import Image 

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pickle

train_folder= '/home/park/workspace/intel-03/chest_xray/chest_xray/train/'
val_folder = '/home/park/workspace/intel-03/chest_xray/chest_xray/val/'
test_folder = '/home/park/workspace/intel-03/chest_xray/chest_xray/test/'

# let's build the CNN model

model = tf.keras.models.Sequential()
#Convolution
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3))) 
# 인풋셰이프 64,64,3지 이란걸 어떻게 알 수 있지
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'],)

# 이중분류라 시그모이드
# 컴파일 loss는 cnn_mnist에서 가져온거라 binary_crossentropy로 바꿔줌

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
#Image normalization.
training_set = train_datagen.flow_from_directory(train_folder, target_size = (64, 64),
                                                 batch_size = 32, class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(val_folder, target_size=(64, 64),
                                                        batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(test_folder, target_size = (64, 64), batch_size = 32,
                                            class_mode = 'binary')
model.summary()

cnn_model = model.fit(training_set, steps_per_epoch=len(test_set), epochs=10, validation_data = validation_generator,
                         validation_steps =len(validation_generator))
# steps_per_epoch란 무엇일까
# validation_generator가 무엇이지?

# 모델 저장, 정확도 출력
test_accu = model.evaluate(test_set,steps= len(validation_generator))
model.save('./HW6_CNN/models/HW06_medical_cnn.h5')
print('The testing accuracy is :',test_accu[1]*100, '%')

# 예측
Y_pred = model.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)


plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validationset'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=False)
plt.clf()

plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'],
loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=False)
plt.clf()

model.save('/home/park/workspace/intel-03/class01/homework/parkwonseok/HW6_CNN/models/HW6_CNN_pneumonia.keras')
with open('history_pneumonia', 'wb') as file_pi: pickle.dump(cnn_model.history, file_pi)