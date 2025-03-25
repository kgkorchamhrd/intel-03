import keras_ocr
import ollama
import matplotlib.pyplot as plt
import cv2
import numpy as np
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

# 한글 폰트 경로 (사용자 환경에 맞게 수정 필요)
font_path = "./font/malgun.ttf"  # 시스템에 맞게 변경
font_prop = fm.FontProperties(fname=font_path)

# Ollama로 번역하는 함수
def use_ollama(prompt):
    response = ollama.chat(model='phi4:14b-q8_0', messages=[
        {'role': 'user', 'content': f"해당 문장을 한국어로 번역해줘! 번역한 문장만 출력해줘: {prompt}"}
    ])
    return getattr(response, "message", {"content": "번역 실패"}).get("content", "번역 실패")

# OCR 텍스트 및 번역 텍스트를 테두리와 함께 그리는 함수
def draw_text_with_outline(draw, position, text, font, text_color, outline_color, outline_thickness = 1):
    x, y = position
    
    # 검은색 테두리 (네 방향으로 offset 적용)
    for dx in [-outline_thickness, 0, outline_thickness]:
        for dy in [-outline_thickness, 0, outline_thickness]:
            if dx == 0 and dy == 0:
                continue  # 원래 위치 제외
            draw.text((x + dx, y + dy), text, font=font, fill=outline_color)

    # 원래 텍스트
    draw.text((x, y), text, font=font, fill=text_color)

# Keras OCR pipeline 초기화
pipeline = keras_ocr.pipeline.Pipeline()

# 이미지 경로 설정
image_name = "exam06"
image_format = ".png"
image_folder = "./test/test_img/"
image_path = image_folder + image_name + image_format
output_path = image_folder + "keras+phi4_" + image_name + ".png"

# 이미지 읽기
original_img = Image.open(image_path)  # 원본 이미지를 PIL로 로드
img = keras_ocr.tools.read(image_path)
img_ocr = original_img.copy()
img_translated = original_img.copy()

# OCR 처리
prediction_groups = pipeline.recognize([img])
ocr_results = prediction_groups[0]

# OCR 결과에서 텍스트 추출
target_texts = [text for text, _ in ocr_results]
ocr_text = "\n".join(target_texts)
print("OCR :", ocr_text)

# Ollama 번역 수행
translated_text = use_ollama(ocr_text)
print("phi4 :", translated_text)

# 번역된 텍스트 전처리 - () 속 부가설명이 나올 시 삭제
translated_text_cleaned = re.sub(r"\(.*?\)", "", translated_text).strip()
translated_lines = translated_text_cleaned.split("\n")

# 이미지 편집을 위한 PIL 객체 생성
draw_ocr = ImageDraw.Draw(img_ocr)
draw_translated = ImageDraw.Draw(img_translated)
font_ocr = ImageFont.truetype(font_path, 20)
font_translation = ImageFont.truetype(font_path, 20)


for i, ((text, box), translated_text) in enumerate(zip(ocr_results, translated_lines)):
    pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(np.array(img_ocr), [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    # 좌측 상단 기준 좌표
    x, y = int(box[0][0]), int(box[0][1])  

    print(f"OCR {i}: '{text}' at ({x}, {y}) → 번역: '{translated_text}'")


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
