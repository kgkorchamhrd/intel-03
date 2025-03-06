import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import pickle

print("Current Working Directory:", os.getcwd())

mainDIR = os.listdir('./hw6_CNN/chest_xray')
print(mainDIR)
train_folder= './hw6_CNN/chest_xray/train/'
val_flder = './hw6_CNN/chest_xray/val/'
test_folder = './hw6_CNN/chest_xray/test/'

os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
print(len(os.listdir(train_n)))

#normal pic
rand_norm = np.random.randint(0, len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ', norm_pic)

norm_pic_address = train_n+norm_pic

#Pneumonia
rand_p = np.random.randint(0, len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p + sic_pic

print('pneumonia picture title:', sic_pic)

#Load the images
norm_load = Image.open(norm_pic_address)
sic_load = Image.open(sic_address)

#plt images
f = plt.figure(figsize = (10,6))
a1 = f.add_subplot(1,2,1)
img_plot = plt.imshow(norm_load)
a1.set_title('Normal')

a2 = f.add_subplot(1,2,2)
img_plot = plt.imshow(sic_load)
a2.set_title('Pneumonia')
plt.show()

num_of_test_samples = 600
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./hw6_CNN/chest_xray/train',
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('./hw6_CNN/chest_xray/test',
                                                        target_size = (64,64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')


#model.add 사용하지 않고 omodel 정의하기
model_in = Input(shape = (64,64, 3))
model = Flatten()(model_in)

model = Dense(activation = 'relu', units = 128)(model)
model = Dense(activation = 'sigmoid', units = 1)(model)

model_fin = Model(inputs=model_in, outputs=model)
model_fin.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


history = model_fin.fit(training_set,
                          steps_per_epoch= 163,
                          epochs = 10,
                          validation_data = validation_generator,
                          validation_steps = 624)

test_accu = model_fin.evaluate(validation_generator, steps=624)

# model_fin.save('medical_ann.h5')
print('The testing accuracy is :', test_accu[1]*100, '%')
Y_pred = model_fin.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
max(y_pred)

model_fin.save('medical_ann.h5')
with open('history_pneumonia', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')

plt.show()

predictions = model_fin.predict(validation_generator)


# Convert predictions to binary (0 = Normal, 1 = Pneumonia)
predicted_labels = (predictions > 0.5).astype(int)

# ===============================
# 🔹 Display 5x10 Grid of Predictions
# ===============================

fig, axes = plt.subplots(5, 10, figsize=(20, 10))

test_images, test_labels = next(validation_generator)  # Get one batch of test images
predictions = model_fin.predict(test_images)

# Convert predictions to binary (0 = Normal, 1 = Pneumonia)
predicted_labels = (predictions > 0.5).astype(int)

# Display 5x10 Grid of Predictions
fig, axes = plt.subplots(5, 10, figsize=(20, 10))

for i, ax in enumerate(axes.flat):
    if i >= len(test_images):
        break
    img = test_images[i]
    pred_label = "Pneumonia" if predicted_labels[i] == 1 else "Normal"
    true_label = "Pneumonia" if test_labels[i] == 1 else "Normal"

    ax.imshow(img)
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10, 
                 color='red' if pred_label != true_label else 'green')
    ax.axis('off')

plt.tight_layout()
plt.show()