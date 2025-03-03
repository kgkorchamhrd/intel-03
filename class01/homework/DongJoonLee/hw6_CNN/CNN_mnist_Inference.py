import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#fashion Mnist
model = tf.keras.models.load_model('./fashion_mnist.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train /255.0, f_image_test /255.0

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Fashion Mnist Prediction, ", np.argmax(predict, axis = 1))


#number Mnist
model = tf.keras.models.load_model('./mnist.h5')
fashion_mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()

f_image_train, f_image_test = f_image_train /255.0, f_image_test /255.0

num = 10
predict = model.predict(f_image_train[:num])
print(f_label_train[:num])
print(" * Mnist Prediction, ", np.argmax(predict, axis = 1))