import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import pickle

img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE # 병렬연산을 할 것인지에 대한 인자를 알아서 처리하도록.

#데이터 셋 나누기 학습, 검증, 테스트
(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                             with_info=True,as_supervised=True,)
# 80%까지 테스트 / 80~90까지 검증데이터 / 90~테스트 데이터
# with_info : 클래스 개수 확인
# as_supervised : 튜플로 변환되어 모델 학습에 사용가능


num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes)

#model = tf.keras.models.load_model('./model/fashion_mnist.h5') 

def prepare(ds, shuffle=False, augment=False):
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
    num_parallel_calls=AUTOTUNE)
    # 전처리 적용
    ds = ds.map(lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE)
    # Batch all datasets
    ds = ds.batch(batch_size)
    # Use data augmentation only on the training set.
    if augment:
        data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                                     layers.RandomRotation(0.2),])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
            # 데이터 로딩과 모델 학습이 병렬로 처리되기 위해
            # prefetch()를 사용해서 현재 배치가 처리되는 동안 다음 배치의 데이터를 미리 로드 하도록 함.
    return ds.prefetch(buffer_size=AUTOTUNE)

# 이 과정으로 학습 최적화된 데이터 반환

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)
#데이터 최적화

# 전이학습 시작
# include_top -> ANN 부분 직접 수정
base_model = tf.keras.applications.MobileNetV3Small( 
weights='imagenet', # Load weights pre-trained on ImageNet.
input_shape = (img_height, img_width, 3), include_top = False)

base_model.trainable = False # 기본 모델의 가중치 동결 학습되지 않게 설정

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
# 추론, 학습에서 다르게 동작하는 layer들을 추론/학습 중 하나로만 동작하게 함.
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_ds, epochs=15, validation_data=val_ds)
model.save('HW6_transfer_learning_flower.keras')
with open('HW6_history_flower', 'wb') as file_pi:pickle.dump(history.history, file_pi)
# history : loss, accuracy, val_loss, valaccuracy 기록 (dic 형태로 저장) -> pickle(.pkl)로 저장
# 이후 학습 과정 데이터를 불러와서 그래프 비교 분석 가능