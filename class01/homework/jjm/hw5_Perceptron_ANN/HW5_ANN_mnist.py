# 250226
# ANN_mnist

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, labe_test) = mnist.load_data()
# normalized iamges
image_train, image_test = image_train / 255.0, image_test / 255.0


class_names = ['Zero', 'One', 'Two', 'Three',
                'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

plt.figure(figsize = (10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_train[i])
    plt.xlabel(class_names[label_train[i]])

plt.savefig('./src_img/mnist_00.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(image_train, label_train, epochs = 10, batch_size = 10)
model.summary()
model.save('./models/mnist.h5')

# 250226
# ANN_fashion_mnist

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_labe_test) = fashion_mnist.load_data()
# normalized iamges
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                'Coat', 'Sandal', 'Shirt', 'Skeaker', 'Bag', 'Ankle boot']

plt.figure(figsize = (10, 10))
for i in range(10):
    plt.subplot(3, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])

plt.savefig('./src_img/fashion_mnist_00.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# fashion ANN
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(f_image_train, f_label_train, epochs = 10, batch_size = 10)
model.summary()
model.save('./models/fashion_mnist.h5')


