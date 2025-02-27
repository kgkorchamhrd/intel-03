import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import pickle

# 데이터 경로 설정
train_folder = './workspace/chest_xray/chest_xray/train/'
val_folder = './workspace/chest_xray/chest_xray/val/'
test_folder = './workspace/chest_xray/chest_xray/test/'

# 데이터셋의 폴더와 이미지 확인
train_n = train_folder + 'NORMAL/'
train_p = train_folder + 'PNEUMONIA/'

# Normal 이미지를 랜덤으로 선택
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
norm_pic_address = train_n + norm_pic

# Pneumonia 이미지를 랜덤으로 선택
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_p]
sic_address = train_p + sic_pic

# 이미지를 로드하여 시각화
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

f = plt.figure(figsize=(10, 6))
a1 = f.add_subplot(1, 2, 1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1, 2, 2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

# 모델 빌드
model_in = Input(shape=(64, 64, 3))
model = Flatten()(model_in)
model = Dense(activation='relu', units=128)(model)
model = Dense(activation='sigmoid', units=1)(model)
model_fin = Model(inputs=model_in, outputs=model)

# 모델 컴파일
model_fin.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 전처리 및 이미지 제너레이터 설정
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('./workspace/chest_xray/chest_xray/train/', target_size=(64, 64), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory('./workspace/chest_xray/chest_xray/val/', target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('./workspace/chest_xray/chest_xray/test', target_size=(64, 64), batch_size=32, class_mode='binary')

model_fin.summary()

# 모델 학습
ann_model = model_fin.fit(training_set, steps_per_epoch=163, epochs=10, validation_data=validation_generator, validation_steps=624)

# 모델 평가
test_accu = model_fin.evaluate(test_set, steps=624)
model_fin.save('medical_ann.h5')
print('The testing accuracy is:', test_accu[1] * 100, '%')

# 예측 수행 (25개 이미지 예측)
Y_pred = model_fin.predict(test_set, 25)
y_pred = np.argmax(Y_pred, axis=1)

# 5x5 이미지로 출력
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
axes = axes.flatten()

# test_set에서 배치 단위로 데이터를 가져오기
for i in range(25):
    img, label = next(test_set)  # next()를 사용하여 배치에서 이미지와 라벨 가져오기
    ax = axes[i]
    ax.imshow(img[0])  # 첫 번째 이미지
    ax.set_title(f'Pred: {y_pred[i]}, True: {label[0]}')  # 예측과 실제값 출력
    ax.axis('off')

plt.tight_layout()

# 이미지를 저장
plt.savefig('predictions_5x5.png')
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
model.save('transfer_learning_pneumonia.keras')
with open('history_pneumonia', 'wb') as file_pi:
    pickle.dump(cnn_model.history, file_pi)
