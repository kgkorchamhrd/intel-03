import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(image_train, label_train), (image_test, label_test) = mnist.load_data()

# 이미지 정규화 (0~1 범위로 스케일링) 및 채널 차원 추가 (흑백 이미지)
image_train, image_test = image_train / 255.0, image_test / 255.0
image_train = np.expand_dims(image_train, axis=-1)  # (28, 28) -> (28, 28, 1)
image_test = np.expand_dims(image_test, axis=-1)

# 클래스 이름 (0~9 숫자)
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 샘플 이미지 출력
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[label_train[i]])
plt.show()

# CNN 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10개 클래스
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(image_train, label_train, epochs=10, batch_size=10)

# 모델 구조 출력 및 저장
model.summary()
model.save('mnist_cnn.h5')

# 모델 로드 및 예측 수행
model = tf.keras.models.load_model('mnist_cnn.h5')

num = 10
predict = model.predict(image_test[:num])

print("정답 레이블:", label_test[:num])
print("예측 결과:", np.argmax(predict, axis=1))

# Fashion-MNIST 데이터셋 로드
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

# 이미지 정규화 및 채널 차원 추가
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
f_image_train = np.expand_dims(f_image_train, axis=-1)  # (28, 28) -> (28, 28, 1)
f_image_test = np.expand_dims(f_image_test, axis=-1)

# 클래스 이름 매핑
f_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 샘플 이미지 출력
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i], cmap=plt.cm.binary)
    plt.xlabel(f_class_names[f_label_train[i]])
plt.show()

# Fashion-MNIST 모델 생성
f_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10개 클래스
])

# 모델 컴파일
f_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
f_model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)

# 모델 구조 출력 및 저장
f_model.summary()
f_model.save('fashion_mnist_cnn.h5')

# 모델 로드 및 예측 수행
f_model = tf.keras.models.load_model('fashion_mnist_cnn.h5')

f_predict = f_model.predict(f_image_test[:num])

print("정답 레이블:", f_label_test[:num])
print("예측 결과:", np.argmax(f_predict, axis=1))
