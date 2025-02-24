from PIL import Image
import numpy as np

image_path = './image.png'
image = Image.open(image_path)

image_array = np.array(image)

image_with_batch = np.expand_dims(image_array, axis=0) 

transposed_image = image_with_batch.transpose(0, 3, 1, 2) 

# Verify the shape of the result
print("Original Shape:", image_array.shape)
print("Shape after adding batch dimension:", image_with_batch.shape)
print("Shape after transposing:", transposed_image.shape)

import cv2

frame = cv2.imread('./image.png'), cv2.COLOR_BGR2RGB
frame = frame/255

print("image shape :", frame.shape)

# dimension 에서 제일 첫번째 값으로 차원 추가
frame2 = np.expand_dims(frame, 0)

# transpose의 필요성: framework마다 요구하는 dimension구조가 다르다
frame3 = frame2.transpose(0, 3, 1, 2)
