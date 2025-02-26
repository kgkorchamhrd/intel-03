import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ==================================================================================================
# mnist
# ==================================================================================================
# mnist 데이터로드
mnist = tf.keras.datasets.mnist

(m_image_train, m_label_train), (m_image_test, f_label_test) = mnist.load_data()
m_image_train, m_image_test = m_image_train / 255.0, m_image_test/255.0

# mnist 데이터확인
plt.figure(figsize=(10, 10))

for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(m_image_train[i])
    plt.xlabel(m_label_train[i])
plt.show()


# 모델 생성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# loss='sparse_categorical_crossentropy' onehot encoding label
# loss='categorical_crossentropy' 단순 label
model.compile(optimizer = 'adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 및 저장.
model.fit(m_image_train, m_label_train, epochs=10, batch_size=10)
model.summary()
model.save('./mnist.h5')




# ==================================================================================================
# fashion-mnist
# ==================================================================================================
# fashion-mnist 데이터 확인
fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, k_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test/255.0

# 라벨
class_names = ['T-shirt/top', 
               'Trouser', 
               'Pullover', 
               'Dress', 
               'Coat',
               'Sandal',
               'Shirt',
               'Sneaker',
               'Bag',
               'Ankle boot']


plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
plt.show()

# 모델 생성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer = 'adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 및 저장.
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('./f-mnist.h5')
