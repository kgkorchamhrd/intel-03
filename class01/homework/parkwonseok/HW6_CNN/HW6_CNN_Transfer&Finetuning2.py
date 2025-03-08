import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
img_height = 255
img_width = 255
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE 

(train_ds, val_ds, test_ds), metadata = tfds.load('tf_flowers',
split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
with_info=True,
as_supervised=True,)

num = 20
def prepare(ds, batch = 1, shuffle=False, augment=False):
            preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
            ds = ds.map(lambda x, y: (tf.image.resize(x, [img_height, img_width]), y),
            num_parallel_calls=AUTOTUNE)
            ds = ds.batch(batch_size)
            return ds.prefetch(buffer_size=AUTOTUNE)

num_classes = metadata.features['label'].num_classes
label_name = metadata.features['label'].names
print(label_name, ", classnum : ", num_classes, ", type: ", type(label_name))
test_ds = prepare(test_ds, num)
image_test, label_test = next(iter(test_ds))
image_test = np.array(image_test)
label_test = np.array(label_test, dtype='int')
# 모델 불러오기
model = tf.keras.models.load_model('./HW6_CNN/models/transfer_learning_flower.keras')
model.summary()
predict = model.predict(image_test)
predicted_classes = np.argmax(predict, axis=1)

print("실제 레이블 | 예측 레이블");
print("------------------------")
for ll in range((label_test.size)):
    print(label_name[label_test[ll]], "|",label_name[predicted_classes[ll]])
print("------------------------")
# print("실제 레이블:", [label_name[idx] for idx in label_test])
# print("예측 레이블:", [label_name[idx] for idx in predicted_classes])
accuracy = np.mean(predicted_classes == label_test)
print(f"정확도: {accuracy:.2%}")