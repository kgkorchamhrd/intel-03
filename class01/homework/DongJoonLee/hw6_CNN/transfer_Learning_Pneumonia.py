import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

img_height = 255
img_width = 255
batch_size = 32
# AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 알아서 처리하도록.
# # Dataset 준비 (https://www.tensorflow.org/tutorials/load_data/images?hl=ko)
# (train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',
#     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#     with_info=True, as_supervised=True,
# )
# num_classes = metadata.features['label'].num_classes
# label_name = metadata.features['label'].names
# print(label_name, ", classnum : ", num_classes)

# def prepare(ds, shuffle=False, augment=False):
#     preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
#     # Resize and rescale all datasets.
#     ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
#     num_parallel_calls=AUTOTUNE)
#     # 전처리 적용
#     ds = ds.map(lambda x, y: (preprocess_input(x), y),
#     num_parallel_calls=AUTOTUNE)
#     # Batch all datasets
#     ds = ds.batch(batch_size)
#     # Use data augmentation only on the training set.
#     if augment:
#         data_augmentation = tf.keras.Sequential([
#         layers.RandomFlip("horizontal_and_vertical"),
#         layers.RandomRotation(0.2),
#         ])
#         ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
#         num_parallel_calls=AUTOTUNE)
#     # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해
#     # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미리 로드 하도록 함.
#     return ds.prefetch(buffer_size=AUTOTUNE)

# train_ds = prepare(train_ds, shuffle=True, augment=True)
# val_ds = prepare(val_ds)
# test_ds = prepare(test_ds)


mainDIR = os.listdir('./hw6_CNN/chest_xray')
print(mainDIR)
train_folder= './hw6_CNN/chest_xray/train/'
val_flder = './hw6_CNN/chest_xray/val/'
test_folder = './hw6_CNN/chest_xray/test/'

os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
print(len(os.listdir(train_n)))

#normal pic
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic

print('pneumonia picture title:', sic_pic)

#Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#plt images
# f = plt.figure(figsize = (10,6))
# a1 = f.add_subplot(1,2,1)
# img_plot = plt.imshow(norm_load)
# a1.set_title('Normal')

# a2 = f.add_subplot(1,2,2)
# img_plot = plt.imshow(sic_load)
# a2.set_title('Pneumonia')
# plt.show()

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./hw6_CNN/chest_xray/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./hw6_CNN/chest_xray/test',
                                                        target_size = (64,64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

train_ds = training_set
val_ds = validation_generator

# include_top -> ANN 부분 직접 수정 (FC부분을 삭제하기)
base_model = tf.keras.applications.MobileNetV3Small(
weights='imagenet', # Load weights pre-trained on ImageNet.
input_shape = (img_height, img_width, 3),
include_top = False)

# 기본 모델의 가중치 동결
base_model.trainable = False

inputs = tf.keras.Input(shape=(img_height, img_width, 3))

model_in = Input(shape = (64,64, 3))
model = Flatten()(model_in)

model = Dense(activation = 'relu', units = 128)(model)
model = Dense(activation = 'sigmoid', units = 1)(model)

model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


history = model_fin.fit(training_set,
                          steps_per_epoch= 163,
                          epochs = 10,
                          validation_data = validation_generator,
                          validation_steps = 624)

test_accu = model_fin.evaluate(validation_generator, steps=624)


# 추론, 학습에서 다르게 동작하는 layer들을 추론/학습 중 하나로만 동작하게 함.
# 아래 부분을 True로 바꾸어주면 Finetuning 
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_ds, epochs=15, validation_data=val_ds)
model.save('transfer_learning_pneumonia.keras')
with open('history_pneumonia', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)