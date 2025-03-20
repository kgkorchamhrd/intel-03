#
# pip install keras-ocr tensorflow==2.10 transformers matplotlib sentencepiece

import keras_ocr
import matplotlib.pyplot as plt
from transformers import T5ForConditionalGeneration, T5Tokenizer

# 1. Keras OCR 모델 로드
pipeline = keras_ocr.pipeline.Pipeline()

# 2. 이미지 로드 및 OCR 수행
image_name = "exam02"
image_format = ".png"
image_folder = "./test/test_img/"
image_path = image_folder + image_name + image_format

output_path = image_folder + "keras+T5_" + image_name + "png"

image = keras_ocr.tools.read(image_path)
prediction_groups = pipeline.recognize([image])

# 3. T5 모델 로드 (영어 -> 한국어 번역)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def translate_text(text: str) -> str:
    input_text = f"translate English to Korean: {text}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(input_ids, max_length=512, num_beams=4, early_stopping=True)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

# 4. OCR 결과에서 텍스트 추출 후 번역
translated_texts = []
for text, box in prediction_groups[0]:
    translated_text = translate_text(text)  # 영어에서 한국어로 번역
    translated_texts.append((translated_text, box))

# 5. 결과 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 원본 이미지 표시
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# OCR 결과 표시
axes[1].imshow(image)
for text, box in prediction_groups[0]:
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    axes[1].plot(x_values + [x_values[0]], y_values + [y_values[0]], 'r-')
    axes[1].text(box[0][0], box[0][1], text, fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.6))
axes[1].set_title("OCR Result")
axes[1].axis('off')

# 번역된 텍스트 표시
axes[2].imshow(image)
for translated_text, box in translated_texts:
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    axes[2].plot(x_values + [x_values[0]], y_values + [y_values[0]], 'r-')
    axes[2].text(box[0][0], box[0][1], translated_text, fontsize=12, color='green', bbox=dict(facecolor='white', alpha=0.6))
axes[2].set_title("Translated Text")
axes[2].axis('off')

# 결과 출력
plt.tight_layout()
# 결과 저장
plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
plt.close()