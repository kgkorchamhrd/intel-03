
# pip install opencv-python matplotlib pytesseract


import easyocr
import matplotlib.pyplot as plt
import cv2

# EasyOCR 리더 초기화
reader = easyocr.Reader(['en']) 

# 이미지 로드
image_name = "exam02"
image_format = ".png"
image_folder = "./test/test_img/"
image_path = image_folder + image_name + image_format

output_path = image_folder + "easyOCR_" + image_name + "png"


# 이미지 읽기
image = cv2.imread(image_path)

# EasyOCR을 사용하여 이미지에서 텍스트 추출
result = reader.readtext(image_path)

# OCR 결과 시각화
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 원본 이미지 출력
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

# 텍스트를 인식한 이미지 출력
image_copy = image.copy()
for (bbox, text, prob) in result:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    # 텍스트 박스 그리기
    cv2.rectangle(image_copy, top_left, bottom_right, (0, 255, 0), 2)
    cv2.putText(image_copy, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
# OCR 결과가 반영된 이미지 출력
ax[1].imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
ax[1].set_title('OCR Result')
ax[1].axis('off')

# 결과 저장
plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
plt.close()

