##GUI추가한거 동작함


import sys
import torch
import easyocr
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QListWidget, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage

class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # YOLOv5 모델 로드
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)  # OCR 모델 로드

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템')
        self.setGeometry(100, 100, 800, 600)

        # 버튼
        self.btn_load = QPushButton('이미지 불러오기', self)
        self.btn_load.clicked.connect(self.loadImage)

        # 이미지 표시 라벨
        self.label_img = QLabel(self)
        self.label_img.setFixedSize(640, 480)

        # 번호판 리스트 출력
        self.list_plates = QListWidget(self)

        # 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        layout.addWidget(self.label_img)
        layout.addWidget(self.list_plates)
        self.setLayout(layout)

    def loadImage(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg)", options=options)

        if file_path:
            plates, processed_img = self.detectLicensePlates(file_path)
            self.displayImage(processed_img)  # 이미지 표시
            self.displayPlates(plates)  # 번호판 리스트 표시

    def detectLicensePlates(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print("이미지를 불러올 수 없습니다.")
            return [], None

        # YOLOv5를 이용한 차량 감지
        results = self.model(img)
        car_detected = []
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.4 and int(cls) == 2:  # 차량 클래스
                x1, y1, x2, y2 = map(int, xyxy)
                car_detected.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # EasyOCR을 이용한 번호판 인식
        THRESHOLD = 0.1
        plates = []
        current_plate = ""

        for bbox, text, conf in self.reader.readtext(img):
            if conf > THRESHOLD:
                for (x1, y1, x2, y2) in car_detected:
                    if x1 < bbox[0][0] < x2 and y1 < bbox[0][1] < y2:  # 차량 내부의 문자만 인식
                        current_plate += text
                        cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 255), 3)

                        if len(current_plate) > 5:
                            plates.append(current_plate)
                            current_plate = ""

        if current_plate:
            plates.append(current_plate)

        return plates, img

    def displayImage(self, img):
        if img is None:
            return

        # OpenCV 이미지 → QImage 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # QLabel에 이미지 설정
        self.label_img.setPixmap(pixmap)
        self.label_img.setScaledContents(True)

    def displayPlates(self, plates):
        self.list_plates.clear()
        for idx, plate in enumerate(plates):
            self.list_plates.addItem(f"차량 {idx + 1} 번호판: {plate}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
