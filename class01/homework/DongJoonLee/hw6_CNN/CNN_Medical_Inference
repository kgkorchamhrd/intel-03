import pickle
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved history
with open('history_pneumonia', 'rb') as file_pi:
    history = pickle.load(file_pi)
# Plot Accuracy and Loss
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train vs Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train vs Validation Loss')

plt.show()

# ===============================
# 🔹 Load Model and Test Data
# ===============================

# Load trained model
model = load_model('./medical_ann.h5')

# Reload validation/test generator
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    './hw6_CNN/chest_xray/test',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # Ensure order is maintained
)

# Load test dataset (1 batch)
test_images, test_labels = next(validation_generator)  # Get one batch of images
predictions = model.predict(test_images)

# Convert predictions to binary (0 = Normal, 1 = Pneumonia)
predicted_labels = (predictions > 0.5).astype(int)

# ===============================
# 🔹 Display 5x10 Grid of Predictions
# ===============================

fig, axes = plt.subplots(5, 10, figsize=(20, 10))

for i, ax in enumerate(axes.flat):
    if i >= len(test_images):
        break
    img = test_images[i]
    pred_label = "Pneumonia" if predicted_labels[i] == 1 else "Normal"
    true_label = "Pneumonia" if test_labels[i] == 1 else "Normal"

    ax.imshow(img)
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=10, color='red' if pred_label != true_label else 'green')
    ax.axis('off')

plt.tight_layout()
plt.show()
