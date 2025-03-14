##yolo5없는거

import easyocr
import cv2
import matplotlib.pyplot as plt

# GPU 대신 CPU 사용
reader = easyocr.Reader(['ko', 'en'], gpu=False)

img_path = 'C:/workspace/mini_project/data/222.jpg'
img = cv2.imread(img_path)

if img is None:
    print("이미지를 열 수 없습니다. 경로를 확인해주세요.")
else:
    plt.figure(figsize=(8, 8))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()

    result = reader.readtext(img)  # img로 변경
    print(result)

    THRESHOLD = 0.1  # 임계값 설정

    plates = []
    current_plate = ""

    for bbox, text, conf in result:
        if conf > THRESHOLD:
            current_plate += text  # 텍스트를 이어붙임
            cv2.rectangle(img, pt1=tuple(map(int, bbox[0])), pt2=tuple(map(int, bbox[2])), color=(0, 255, 0), thickness=3)

            # 번호판의 끝을 판단할 기준
            if len(current_plate) > 5:  # 예시 기준: 번호판 길이가 5 이상이면 분리
                plates.append(current_plate)
                current_plate = ""

    # 마지막 번호판 추가
    if current_plate:
        plates.append(current_plate)

    # 차량 번호판 출력
    for idx, plate in enumerate(plates):
        print(f"차량 {idx+1} 번호판: {plate}")

    # 이미지에 번호판 영역 표시 후 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()
