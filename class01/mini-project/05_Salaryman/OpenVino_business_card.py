import cv2
import numpy as np
import pytesseract
import openvino as ov
from pathlib import Path
import re

# 모델 디렉토리 및 이름 설정
base_model_dir = Path("./hellow-detection/model").expanduser()
model_name = "horizontal-text-detection-0001"
model_xml_name = f"{model_name}.xml"
model_bin_name = f"{model_name}.bin"

# 모델 경로 설정
model_xml_path = base_model_dir / model_xml_name
model_bin_path = base_model_dir / model_bin_name

# 모델 다운로드 확인
if not model_xml_path.exists():
    print(f"{model_name} 모델을 다운로드해야 합니다.")
    exit()
else:
    print(f"{model_name} 모델이 이미 {base_model_dir}에 다운로드 되어 있습니다.")

# 장치 선택
device = "CPU"  # 장치 선택을 간단히 설정 (예: "CPU", "GPU")

# OpenVINO 코어 초기화
core = ov.Core()

# 모델 읽기 및 컴파일
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device)

# 입력 및 출력 레이어 정의
input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output("boxes")

# 결과를 이미지로 변환하는 함수
def convert_result_to_image_with_text_labels(bgr_image, resized_image, boxes, threshold=0.3):
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    (real_y, real_x), (resized_y, resized_x) = (
        bgr_image.shape[:2],
        resized_image.shape[:2],
    )
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # BGR 이미지를 RGB로 변환
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    detected_texts = []  # 추출된 텍스트를 저장할 리스트

    # 박스 내 텍스트 추출 및 이미지에 그리기
    for box in boxes:
        conf = box[-1]
        if conf > threshold:
            (x_min, y_min, x_max, y_max) = [
                (int(max(corner_position * ratio_y, 10)) if idx % 2 else int(corner_position * ratio_x)) for idx, corner_position in enumerate(box[:-1])
            ]

            # 박스 그리기
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # 박스 내 텍스트 추출 (OCR)
            roi = bgr_image[y_min:y_max, x_min:x_max]
            text = pytesseract.image_to_string(roi, config='--psm 6')

            if text.strip():
                rgb_image = cv2.putText(
                    rgb_image,
                    text.strip(),
                    (x_min, max(y_min - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    2,
                    cv2.LINE_AA,
                )

                # 텍스트 저장
                detected_texts.append((text.strip(), x_min, x_max))  # 텍스트와 x_min, x_max 좌표 저장

    return rgb_image, detected_texts

# 한국 성씨 목록 (영문)
last_names = ["Kim", "Lee", "Park", "Choi", "Cho", "Kang", "Jung", "Im", "Yoon", "Jang", "Oh", "Seo", "Ryu", "Ahn", "Hong", "Bae", "Lim", "Son", "Yang", "Chun"]

# 텍스트에서 이름, 전화번호, 이메일 추출 함수
def extract_information(text):
    # 정규식 패턴 정의
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,3}\)?[-.\s]?)?(\d{3,4})[-.\s]?(\d{4})'
    name_pattern = r'[A-Za-z]+'

    # 정규식으로 정보 추출
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    names = re.findall(name_pattern, text)

    # 이름을 성씨 기준으로 추출
    detected_names = []
    for name in names:
        for last_name in last_names:
            if name.startswith(last_name):
                detected_names.append(name)

    return detected_names, phones, emails

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from camera.")
        break

    # 이미지 크기 조정
    N, C, H, W = input_layer_ir.shape
    resized_frame = cv2.resize(frame, (W, H))

    # 네트워크 입력 크기에 맞게 형태 변환
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

    # 실시간 카메라 화면 보여주기
    cv2.imshow("Camera Feed", frame)

    # 'c' 키를 누르면 이미지 캡처 후 처리
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # 'c'가 눌리면 모델에 입력하고 결과 처리
        boxes = compiled_model([input_frame])[output_layer_ir]

        # 제로값을 가지는 박스 제거
        boxes = boxes[~np.all(boxes == 0, axis=1)]

        # 결과 이미지와 텍스트 라벨 생성
        result_frame, detected_texts = convert_result_to_image_with_text_labels(frame, resized_frame, boxes)

        # 결과 이미지 저장
        output_image_path = "./hellow-detection/output_detected_image_from_camera.jpg"
        cv2.imwrite(output_image_path, cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))  # BGR 형식으로 저장
        print(f"Detected image with text saved to {output_image_path}")

        # **detected_texts를 텍스트 파일로 저장하는 부분 추가**
        if detected_texts:  # 텍스트가 있을 때만 저장
            extracted_text_path = "./hellow-detection/extracted_text.txt"
            with open(extracted_text_path, 'w', encoding='utf-8') as f:
                for item in detected_texts:
                    f.write(f"{item}\n")  # 각각의 텍스트를 새 줄에 저장
            print(f"Detected text list saved to {extracted_text_path}")

        # 텍스트에서 이름, 전화번호, 이메일 추출
        names, phones, emails = extract_information(" ".join([item[0] for item in detected_texts]))  # 텍스트 합쳐서 전달
        print(f"Extracted Names: {names}")  # 디버깅용 출력
        print(f"Extracted Phones: {phones}")  # 디버깅용 출력
        print(f"Extracted Emails: {emails}")  # 디버깅용 출력

        # 추출된 전화번호를 '-' 구분자로 연결하여 저장
        formatted_phones = []
        for phone in phones:
            phone_str = ' '.join([p.replace('-', '') for p in phone if p])  # '-' 제거 후 공백 추가
            phone_str = phone_str.replace(' ', '-')  # 모든 공백을 '-'로 변환
            formatted_phones.append(phone_str)

        # 추출된 전화번호, 이름, 이메일을 한 파일에 저장ㅂ
        if formatted_phones or names or emails:
            output_txt_path = "./hellow-detection/extracted_info.txt"
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                if formatted_phones:
                    f.write("Phones:")
                    for phone in formatted_phones:
                        f.write(f"{phone}\n")
                if names:
                    f.write("Names:")
                    for name in names:
                        f.write(f"{name}\n")
                if emails:
                    f.write("Emails:")
                    for email in emails:
                        f.write(f"{email}\n")
            print(f"Extracted information saved to {output_txt_path}")

        # 결과 이미지 표시
        cv2.imshow("Processed Image", result_frame)

    # 'q' 키를 눌렀을 때 바로 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
