import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import pickle

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input , BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder='./chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'

os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'

print(len(os.listdir(train_n)))
rand_norm = np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic

rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p+sic_pic
print('pneumonia picture title: ', sic_pic)

norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize= (10, 6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a1 = f.add_subplot(1,2,2)
img_plot = plt.imshow(sic_load)
a1.set_title('Pnuemonia')
plt.show()



#CNN
# Define the CNN model
model_fin = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])


num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./chest_xray/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./chest_xray/val/',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

test_set = test_datagen.flow_from_directory('./chest_xray/test',
                                          target_size = (64, 64),
                                          batch_size = 32,
                                          class_mode = 'binary')

model_fin.summary()

# fit(training)
# Compile the model

model_fin.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

cnn_model = model_fin.fit(training_set,epochs=10, batch_size=32, validation_data=validation_generator)

test_accu = model_fin.evaluate(test_set, steps=624)

model_fin.save('medical_cnn.h5')
with open('history_pn', 'wb') as f:
    pickle.dump(cnn_model.history, f)
print('The testing accuracy is :', test_accu[1]*100, '%')
Y_pred = model_fin.predict(test_set, 100)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

# Show model summary
model_fin.summary()




# Accuracy
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show(block=True)
plt.clf()

#Loss
plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show(block=True)
plt.clf()
