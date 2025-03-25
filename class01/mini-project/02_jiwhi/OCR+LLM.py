
# 사용하는 패키지 목록
# pip install tensorflow==2.10 keras-ocr ollama matplotlib opencv-python re

import keras_ocr
import ollama
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm


# 폰트 경로
font_path = "./font/malgun.ttf"
font_prop = fm.FontProperties(fname = font_path)

# 이미지 경로 
image_name = "exam04"
image_format = ".png"
image_folder = "./test_image/"
result_folder = "./result/"
image_path = image_folder + image_name + image_format
output_path = result_folder + "Transrate_" + image_name + ".png"


# Ollama python API를 사용하는 함수
def use_ollama(prompt):
    response = ollama.chat(model = 'phi4:14b-q8_0', messages = [
        {'role': 'user', 'content': f"해당 문장을 한국어로 번역해줘! 번역한 문장만 출력해줘: {prompt}"}
    ])
    return getattr(response, "message", {"content": "번역 실패"}).get("content", "번역 실패")


# 텍스트를 여러개 겹쳐서 테두리를 만드는 함수
def draw_text_with_outline(draw, position, text, font, text_color, outline_color, outline_thickness = 1):
    x, y = position
    
    # 상하좌우로 오프셋
    for dx in [-outline_thickness, 0, outline_thickness]:
        for dy in [-outline_thickness, 0, outline_thickness]:
            if dx == 0 and dy == 0:
                continue                  # 정위치 좌표는 제외처리
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)


# Keras OCR 초기화
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지 읽기
original_img = Image.open(image_path)
img = keras_ocr.tools.read(image_path)
img_ocr = original_img.copy()
img_translated = original_img.copy()

# OCR 처리
prediction_groups = pipeline.recognize([img])
ocr_results = prediction_groups[0]
target_texts = [text for text, _ in ocr_results]
ocr_text = "\n".join(target_texts)
print("OCR :", ocr_text)

# Ollama 번역
translated_text = use_ollama(ocr_text)
print("phi4 :", translated_text)
translated_text_cleaned = re.sub(r"\(.*?\)", "", translated_text).strip()
translated_lines = translated_text_cleaned.split("\n")

# 이미지 편집을 위한 PIL 객체 생성
draw_ocr = ImageDraw.Draw(img_ocr)
draw_translated = ImageDraw.Draw(img_translated)
font_ocr = ImageFont.truetype(font_path, 20)
font_translation = ImageFont.truetype(font_path, 20)


# 텍스트 표시
for (text, box), translated_text in zip(ocr_results, translated_lines):
    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(np.array(img_ocr), [pts], isClosed=True, color=(0, 255, 0), thickness = 1)

    x, y = int(box[0][0]), int(box[0][1])  
    print(f"OCR: '{text}' at ({x}, {y}) → 번역: '{translated_text}'")

    # OCR 텍스트 출력 - 초록색
    draw_text_with_outline(draw_ocr, (x, y - 25), text, font_ocr, text_color=(0, 255, 0), outline_color=(0, 0, 0))

    # 번역 텍스트 출력 - 빨간색
    draw_text_with_outline(draw_translated, (x, y - 25), translated_text, font_translation, text_color=(255, 0, 0), outline_color=(0, 0, 0))

# 결과 출력
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본
axes[0].imshow(original_img)
axes[0].set_title("Original")
axes[0].axis('off')

# OCR
axes[1].imshow(img_ocr)
axes[1].set_title("OCR")
axes[1].axis('off')

# 번역
axes[2].imshow(img_translated)
axes[2].set_title("Translate")
axes[2].axis('off')

# 결과 저장
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
