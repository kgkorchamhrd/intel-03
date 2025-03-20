import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# 경로 설정
train_folder = os.path.expanduser('~/chest_xray/train/')
val_folder = os.path.expanduser('~/chest_xray/val/')
test_folder = os.path.expanduser('~/chest_xray/test/')

# 데이터 전처리
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터셋
training_set = train_datagen.flow_from_directory(
    train_folder,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 검증 데이터셋
validation_generator = test_datagen.flow_from_directory(
    val_folder,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# 모델 정의 (ANN 모델)
model = Sequential()

# 입력 이미지를 1D 벡터로 펼침 (Flatten)
model.add(Flatten(input_shape=(64, 64, 3)))  # 64x64 크기의 이미지

# Fully Connected Layer (Dense Layer)
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 이진 분류

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set) // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator) // 32
)

# 학습된 모델 저장
model.save(os.path.expanduser('~/chest_xray_ann_model.keras'))  # Keras 포맷으로 저장

# 1. 모델 성능 분석 그래프
# 정확도와 손실 그래프를 그립니다.
plt.figure(figsize=(12, 6))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 그래프를 파일로 저장
plt.savefig('training_performance.png')  # 그래프를 파일로 저장합니다
print("Performance graphs saved as 'training_performance.png'.")

# 2. 예측 코드
# 여러 이미지를 모델에 입력하여 예측을 수행하고, 이미지와 예측 결과를 하나의 그림에 표시
def save_predicted_images_with_graph(test_folder, save_path='predicted_results.png'):
    # 이미지 파일 목록 가져오기
    images = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder) if fname.endswith('.jpeg') or fname.endswith('.png')]

    # 플롯 생성
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))  # 5x4 그리드 (20개 이미지 표시)
    axes = axes.flatten()

    for i, image_path in enumerate(images[:20]):  # 첫 20개 이미지를 처리
        img = load_img(image_path, target_size=(64, 64))
        img_array = np.array(img) / 255.0  # 정규화
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

        # 예측
        prediction = model.predict(img_array)

        # 예측 결과 출력
        prediction_text = f"{'Pneumonia' if prediction >= 0.5 else 'Normal'}\n{prediction[0][0]:.4f}"

        # 이미지와 예측 결과 그리기
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Prediction: {prediction_text}')
        ax.axis('off')  # 축 숨기기

    plt.tight_layout()

    # 그래프와 함께 이미지 결과를 저장
    plt.savefig(save_path, bbox_inches='tight')  # 예측 결과와 그래프가 포함된 이미지 저장
    print(f"Predicted images and results saved as '{save_path}'")

# 예시로 'test/NORMAL' 폴더에서 예측 결과를 저장하는 함수 호출
save_predicted_images_with_graph('/home/hyowon/chest_xray/test/NORMAL', 'predicted_results.png')
