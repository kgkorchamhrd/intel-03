import cv2
import numpy as np

frame = cv.imread('./lena.png', cv.COLOR_BGR2RGB)
# ( 배치 추가필요, H, W, C)
frame = frame/255 # normaliztion 정규화

print("lena shape:", frame.shape) # 차원이 몇인지 나옴
frame2 = np.expend_dims(frame, 0) # 0번째 자리에 들어간다 (B, H, W, C)
                                  # 오픈비노는 채널이 앞에 나와있다 
frame3 = frame2.tranpose(0, 3, 1, 2) # (B, H ,W ,C) -> (B, C, H, W) 