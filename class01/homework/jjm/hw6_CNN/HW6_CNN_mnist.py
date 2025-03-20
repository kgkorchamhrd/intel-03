
# 2450303 CNN mnist

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist

(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
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
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
fit_hist = model.fit(f_image_train, f_label_train, epochs = 10, batch_size = 10, validation_data=(f_image_test, f_label_test))
model.summary()
model.save('./models/fashion_mnist.h5')


score = model.evaluate(f_image_test, f_label_test, verbose = 0)
print('Final test set accuracy', score[1])


plt.plot(fit_hist.history['accuracy'], label='Train Accuracy')  
plt.plot(fit_hist.history['val_accuracy'], label='Validation Accuracy')  
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()  
plt.savefig('./src_img/fashion_mnist_01_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

model = load_model('./models/fashion_mnist.h5')

my_sample = np.random.randint(10000)
print("Actual Label:", class_names[f_label_test[my_sample]])
input_img = f_image_test[my_sample].reshape(1, 28, 28, 1)

pred = model.predict(input_img)
predicted_class = np.argmax(pred)

print("Predicted Label:", class_names[predicted_class])
plt.title("Pred : {}".format(class_names[predicted_class]))
plt.axis('off')
plt.imshow(f_image_test[my_sample], cmap='gray')
plt.savefig('./src_img/fashion_mnist_02_my_sample.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()