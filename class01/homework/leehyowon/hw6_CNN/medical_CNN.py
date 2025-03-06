import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator

# 경로 설정
test_folder = os.path.expanduser('~/chest_xray/test/')

# 모델 로드 (이미 학습된 모델을 사용)
model = tf.keras.models.load_model(os.path.expanduser('~/chest_xray_ann_model.h5'))

# 훈련을 진행한 후 history 객체가 생성됩니다.
# 모델 훈련을 위한 데이터셋을 불러오는 코드 (이미 학습된 모델이 아니면 여기에 모델 훈련 코드를 추가하세요)
train_folder = os.path.expanduser('~/chest_xray/train/')
val_folder = os.path.expanduser('~/chest_xray/val/')

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
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 훈련 (history 객체를 받습니다)
history = model.fit(
    training_set,
    steps_per_epoch=len(training_set) // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator) // 32
)

# 훈련 후 모델 저장
model.save(os.path.expanduser('~/chest_xray_ann_model.h5'))

# 예측 이미지 결과를 저장하는 함수
def save_predicted_images_with_graph(image_folder, save_path, num_images=20):
    image_files = os.listdir(image_folder)[:num_images]
    
    # 한 이미지를 저장할 figure를 준비 (5x4 grid) + 1 그래프
    fig, axes = plt.subplots(6, 4, figsize=(15, 20))  # 5x4 그리드 + 1 줄로 성능 그래프 추가
    axes = axes.ravel()  # Flatten the 2D array of axes to 1D for easier iteration
    
    # 각 이미지에 대해 예측 수행
    for i, image_file in enumerate(image_files):
        img_path = os.path.join(image_folder, image_file)
        img = load_img(img_path, target_size=(64, 64))
        img_array = np.array(img) / 255.0  # 정규화
        img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

        # 예측
        prediction = model.predict(img_array)
        
        # 예측된 클래스 결정
        if prediction >= 0.5:
            label = f"Pneumonia ({prediction[0][0]*100:.2f}%)"
        else:
            label = f"Normal ({(1 - prediction[0][0])*100:.2f}%)"

        # 이미지와 예측 결과 표시
        axes[i].imshow(img)
        axes[i].axis('off')  # 축 제거
        axes[i].set_title(label, fontsize=10)
    
    # 성능 그래프 추가
    # 정확도 그래프
    axes[20].plot(history.history['accuracy'], label='Train Accuracy')
    axes[20].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[20].set_title('Accuracy')
    axes[20].set_xlabel('Epochs')
    axes[20].set_ylabel('Accuracy')
    axes[20].legend()

    # 손실 그래프
    axes[21].plot(history.history['loss'], label='Train Loss')
    axes[21].plot(history.history['val_loss'], label='Validation Loss')
    axes[21].set_title('Loss')
    axes[21].set_xlabel('Epochs')
    axes[21].set_ylabel('Loss')
    axes[21].legend()

    # 결과 이미지를 저장
    plt.tight_layout()  # 자동으로 배치 조정
    plt.savefig(save_path)
    plt.close()

    print(f"Predicted image grid and performance graphs saved at {save_path}")

# 예시로 'test/NORMAL' 폴더의 이미지 20개에 대해 예측 결과와 그래프를 저장
save_predictions_and_graphs_as_images = '/home/hyowon/chest_xray/test_results/predicted_grid_with_graphs.png'
save_predicted_images_with_graph('/home/hyowon/chest_xray/test/NORMAL', save_predictions_and_graphs_as_images)
