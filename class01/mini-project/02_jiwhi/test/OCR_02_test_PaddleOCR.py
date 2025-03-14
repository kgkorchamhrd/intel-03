
# pip install opencv-python matplotlib paddleocr paddlepaddle protobuf>=3.20.2 (if preinstalled tensorflow)


import cv2
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# PaddleOCR 모델 로드
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# 'ch', 'ch_doc', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari'

# OCR 실행할 이미지 경로
image_name = "exam02"
image_format = ".png"
image_folder = "./test/test_img/"
image_path = image_folder + image_name + image_format

output_path = image_folder + "paddleocr_" + image_name + "png"

# 이미지 로드
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR, Matplotlib은 RGB

# OCR 실행
results = ocr.ocr(image_path, cls=True)

# OCR 결과를 적용한 이미지 복사본 생성
image_with_ocr = image.copy()

# OCR 결과 이미지에 표시
for result in results:
    for res in result:
        box, (text, score) = res[0], res[1]

        # 박스 좌표 가져오기
        x_min, y_min = map(int, box[0])
        x_max, y_max = map(int, box[2])

        # Bounding Box 그리기 (빨간색)
        cv2.rectangle(image_with_ocr, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        # OCR 텍스트 추가 (파란색)
        cv2.putText(image_with_ocr, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Matplotlib을 사용하여 원본 & OCR 결과 subplot 생성
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(image_with_ocr)
axes[1].set_title("OCR Result")
axes[1].axis("off")

# 결과 이미지 저장
plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
plt.show()

print(f"OCR 결과 subplot 이미지 저장 완료: {output_path}")
