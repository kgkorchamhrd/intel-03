# exam04
# 1. 임의의 이미지 파일을 불러온다

# 2. Numpy의expend_dims를 사용해서이미지파일의차원을하나 더늘려(Height, Width, Channel)을 
#    (Batch, Height, Width, Channel)로 확장한다. (이미지 출력 불필요)

# 3. Numpy의transpose를 이용해서차원의순서를(Batch, Width, Height, Channel)
#    에서 (Batch, channel, width, height) 로 변경한다. (이미지출력불필요)


import cv2
import numpy as np

frame = cv2.imread('./src_img/gan_predict01.png', cv2.COLOR_BAYER_BG2BGR)
frame = frame / 255

print("image shape : ", frame.shape)
frame = np.expand_dims(frame, 0)            # 맨앞 0번 자리에 차원추가

frame = np.transpose(0, 3, 1, 2)



# import numpy as np
# from PIL import Image

# img_path = './src_img/gan_predict01.png'


# img = Image.open(img_path)

# img = np.array(img)
# print(img.shape)

# img_expand = np.expand_dims(img, axis = 0)
# print(img_expand.shape)

# img_trans = np.transpose(img_expand)
# print(img_trans.shape)
