## gui없는 웹캠동작

import torch
import cv2
import numpy as np

# YOLOv5 모델 로딩 (이미 사전 훈련된 모델 사용)
model = torch.hub.load('ultralytics/yolov5',
                       'yolov5s')  # 'yolov5s' 모델은 빠르지만 정확도가 낮을 수 있음, 'yolov5m', 'yolov5l' 등의 모델로 성능 개선 가능

# 웹캠을 사용하여 실시간 비디오 캡처 시작
cap = cv2.VideoCapture(0)  # 기본 웹캠 (0번)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 프레임 캡처
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLOv5 모델로 객체 감지 (번호판 포함)
    results = model(frame)

    # 감지된 객체의 좌표와 라벨 추출
    # results.xyxy[0] -> x1, y1, x2, y2, confidence, class
    # 번호판 클래스는 모델이 감지한 클래스 중 하나로 지정되어 있어야 함 (일반적으로 'car' 또는 'plate' 등)
    for *xyxy, conf, cls in results.xyxy[0]:
        if conf > 0.4:  # 확신도가 일정 이상인 객체만 처리
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 번호판을 표시하는 사각형

            # 감지된 번호판 클래스 라벨 (class label이 번호판에 해당하는 클래스여야 함)
            label = results.names[int(cls)]
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 실시간 영상 창에 번호판 영역 표시
    cv2.imshow('Car Plate Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
