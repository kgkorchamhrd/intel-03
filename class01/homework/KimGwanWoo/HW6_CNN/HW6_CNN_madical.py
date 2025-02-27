import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import pickle

# 데이터 경로 설정
train_folder = './workspace/chest_xray/chest_xray/train/'
val_folder = './workspace/chest_xray/chest_xray/val/'
test_folder = './workspace/chest_xray/chest_xray/test/'

# CNN 모델 빌드
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 전처리 및 이미지 제너레이터 설정
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_folder, target_size=(64, 64), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(val_folder, target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(test_folder, target_size=(64, 64), batch_size=32, class_mode='binary')

model.summary()

# 모델 학습
cnn_model = model.fit(training_set, steps_per_epoch=len(training_set), epochs=10, validation_data=validation_generator, validation_steps=len(validation_generator))

# 모델 평가
test_accu = model.evaluate(test_set, steps=len(test_set))
model.save('medical_cnn.h5')
print('The testing accuracy is:', test_accu[1] * 100, '%')

# 예측 수행 (25개 이미지 예측)
Y_pred = model.predict(test_set, 25)
y_pred = (Y_pred > 0.5).astype(int)  # 이진 분류이므로 0 또는 1로 변환

# 5x5 이미지로 출력
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.flatten()

# test_set에서 배치 단위로 데이터를 가져오기
for i in range(25):
    img, label = next(test_set)  # next()를 사용하여 배치에서 이미지와 라벨 가져오기
    ax = axes[i]
    ax.imshow(img[0])  # 첫 번째 이미지
    ax.set_title(f'Pred: {y_pred[i][0]}, True: {label[0]}')  # 예측과 실제값 출력
    ax.axis('off')

plt.tight_layout()
plt.savefig('predictions_5x5.png')  # 이미지 저장
plt.show()
plt.clf()

# Loss 및 Accuracy 그래프 출력
plt.plot(cnn_model.history['accuracy'])
plt.plot(cnn_model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_accuracy.png')
plt.show()
plt.clf()

plt.plot(cnn_model.history['val_loss'])
plt.plot(cnn_model.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.savefig('train_loss.png')
plt.show()
plt.clf()

# 모델 저장 및 학습 기록 저장
model.save('cnn_pneumonia.keras')
with open('history_pneumonia', 'wb') as file_pi:
    pickle.dump(cnn_model.history, file_pi)
