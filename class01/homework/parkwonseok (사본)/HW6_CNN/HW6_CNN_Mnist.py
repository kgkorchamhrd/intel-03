import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train),(f_image_test, f_lable_test) = fashion_mnist.load_data()

#training할 이미지셋과 test할 이미지 셋 255로 색상값 나눔
f_image_train, f_image_test = f_image_train/255.0,f_image_test/255.0

# class number mapping 레이블은 숫자로 저장 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])

plt.show()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) # 인풋셰이프가 첫번째 컨볼루션에 바로들어감감
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3),activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'],)
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)

model.save('./models/HW6_CNN_fashion_mnist.h5')

model = tf.keras.models.load_model('./models/HW6_fashion_mnist.h5')

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))