import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test,
                                 f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
# plt.show()

model = tf.keras.Sequential()  # 순서대로 모델을 쌓아라
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))  # dense 퍼셉트론의 의미층
model.add(tf.keras.layers.Dense(64, activation='relu'))
# 숫자 0부터 9까지 10개의 클래스
model.add(tf.keras.layers.Dense(10, activation='softmax'))
# softmax란? 우리가 0~9까지 있으면 5가 정답이면 확률이 높은것을 출력하는함수
model.summary()

# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     # sparse? -> 3일때 -> 로 컴파일해줌 [0,0,1,0,0,0,0,0,0,0]
#     metrics=['accuracy'],
# )
# model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
