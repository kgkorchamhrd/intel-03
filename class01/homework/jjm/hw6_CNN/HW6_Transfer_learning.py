import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_folder = './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'


base_model = tf.keras.applications.MobileNetV3Small(weights='imagenet', 
                                                    input_shape=(64, 64, 3), 
                                                    include_top=False)
base_model.trainable = False  


inputs = tf.keras.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)  
x = tf.keras.layers.GlobalAveragePooling2D()(x)  
x = tf.keras.layers.Dropout(0.2)(x)  
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  


model_fin = tf.keras.Model(inputs, outputs)


model_fin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_folder, target_size=(64, 64), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(val_folder, target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(test_folder, target_size=(64, 64), batch_size=32, class_mode='binary')

cnn_model = model_fin.fit(training_set, steps_per_epoch=163, epochs=10, validation_data=validation_generator, validation_steps=624)

test_accu = model_fin.evaluate(test_set, steps=624)
print('The testing accuracy is :', test_accu[1] * 100, '%')

model_fin.save('./models/PNEUMONIA_TRANSFER.h5')


plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('./src_img/TL_PNEUMONIA_01_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()


plt.plot(cnn_model.history['loss'])
plt.plot(cnn_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('./src_img/TL_PNEUMONIA_02_loss.png', dpi=300, bbox_inches='tight')
plt.close()

