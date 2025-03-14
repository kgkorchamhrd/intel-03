## 인식 못한다고 뜸

import sys
import easyocr
import cv2
import numpy as np
from openvino.runtime import Core
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QListWidget, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage


class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

        # OpenVINO Inference Engine 초기화
        self.ie = Core()
        self.model = self.ie.read_model(model="C:/workspace/mini_project/yolov5_openvino/yolov5su.xml")
        self.compiled_model = self.ie.compile_model(model=self.model, device_name="CPU")

        # EasyOCR 모델 로드
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템 (OpenVINO)')
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
        print(f"이미지 로드 중: {img_path}")

        # 한글 경로 문제 해결을 위한 파일 로드 방식 변경
        image_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"이미지를 불러올 수 없습니다: {img_path}")
            return [], None

        print("이미지 로드 성공")

        # YOLOv5 OpenVINO 모델로 차량 감지
        input_blob = next(iter(self.compiled_model.inputs))
        output_blob = next(iter(self.compiled_model.outputs))

        img_resized = cv2.resize(img, (640, 640))  # YOLOv5 입력 크기로 조정
        img_transposed = img_resized.transpose((2, 0, 1))  # 채널 변경
        img_input = np.expand_dims(img_transposed, axis=0) / 255.0  # 정규화

        results = self.compiled_model([img_input])[output_blob]  # 추론 실행
        print(f"OpenVINO 모델 출력 크기: {results.shape}")

        # 차량 인식 결과를 그리기 위한 리스트
        car_detected = []
        for detection in results[0]:  # YOLOv5 결과 처리
            print(f"추론 결과: {detection}")  # 결과 로그 출력
            x1, y1, x2, y2, conf, cls = detection[:6]
            if conf > 0.4 and int(cls) == 2:  # 차량 클래스 (YOLO COCO 데이터셋 기준)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                car_detected.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # 차량 영역 상자
                cv2.putText(img, 'car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # EasyOCR을 이용한 번호판 인식
        THRESHOLD = 0.1  # 신뢰도 기준을 좀 더 높임
        plates = []
        current_plate = ""


        print("번호판 인식 시작")
        for bbox, text, conf in self.reader.readtext(img):
            print(f"번호판 후보: {text}, 신뢰도: {conf}")
            if conf > THRESHOLD:  # 신뢰도가 0.5 이상인 경우
                # 차량 감지된 영역과 번호판을 매칭
                for (x1, y1, x2, y2) in car_detected:
                    if x1 < bbox[0][0] < x2 and y1 < bbox[0][1] < y2:  # 차량 내부의 문자만 인식
                        plates.append(text)
                        # 번호판 위치 상자 그리기
                        cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 255), 3)

        if not plates:
            print("번호판을 인식할 수 없습니다.")

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
        if plates:
            for idx, plate in enumerate(plates):
                self.list_plates.addItem(f"차량 {idx + 1} 번호판: {plate}")

        else:
            self.list_plates.addItem("번호판을 인식할 수 없습니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
