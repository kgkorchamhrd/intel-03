import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import cv2

fashion_mnist=tf.keras.datasets.fashion_mnist
#mnist = tf.keras.datasets.mnist
 
(f_image_train, f_label_train),(f_image_test,f_label_test)=fashion_mnist.load_data()
f_image_train,f_image_test = f_image_train/255.0,f_image_test/255.0

class_names = ['Tshirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shhirt','Sneaker','Bag','Ankel boot']

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
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')

model = tf.keras.models.load_model('./mnist.h5')
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = f_image_train / 255.0, f_image_test / 255.0

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Prediction, ", np.argmax(predict, axis = 1))