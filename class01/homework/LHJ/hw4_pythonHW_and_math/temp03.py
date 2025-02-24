import cv2
import numpy as np

frame = cv2.imread('img01.jpg', cv2.COLOR_BGR2RGB)
frame = frame/255

print("img shape : ", frame.shape)

# (H,W,C) 에서 H 앞에 차원 추가
frame2 = np.expand_dims(frame, 0)

frame3 = frame2.transpose(0, 3, 1, 2)
