import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('./models/PNEUMONIA_ANN.h5')

mainDIR = os.listdir('./chest_xray')
print("DATA-load : ", mainDIR)
test_folder = './chest_xray/test/'
test_n = test_folder + 'NORMAL/'
test_p = test_folder + 'PNEUMONIA/'


num_images = 25
normal_images = np.random.choice(os.listdir(test_n), size=12, replace=False)
pneumonia_images = np.random.choice(os.listdir(test_p), size=13, replace=False)


image_paths = [test_n + img for img in normal_images] + [test_p + img for img in pneumonia_images]
labels = ['NORMAL'] * 12 + ['PNEUMONIA'] * 13


fig, axes = plt.subplots(5, 5, figsize=(12, 12))
fig.suptitle("Pneumonia predict ANN", fontsize=16)


for i, ax in enumerate(axes.flat):
    img = Image.open(image_paths[i])
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized.convert('RGB')) / 255.0

    prediction = model.predict(np.expand_dims(img_array, axis=0))
    pred_label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"

    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"Label target : {labels[i]}\n Label predict : {pred_label}", fontsize=10)

plt.tight_layout()
plt.savefig('./src_img/ANN_PNEUMONIA_04_predict.png.png', dpi=300, bbox_inches='tight')
plt.show()

print("prediect Complete!")
