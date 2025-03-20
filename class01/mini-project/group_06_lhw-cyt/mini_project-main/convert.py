import sys
import torch
import easyocr
import cv2
import numpy as np
import re
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal, Qt

class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)
        
        self.cap = cv2.VideoCapture(5)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.thread = VideoCaptureThread(self.cap)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        self.captured_image = None

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템')
        self.setGeometry(100, 100, 1280, 720)
        self.setFocusPolicy(Qt.StrongFocus)  # 🔹 키 입력을 받을 수 있도록 설정

        self.label_live = QLabel(self)
        self.label_live.setFixedSize(640, 480)
        self.label_live.setAlignment(Qt.AlignCenter)

        self.label_captured = QLabel(self)
        self.label_captured.setFixedSize(640, 480)
        self.label_captured.setAlignment(Qt.AlignCenter)

        self.list_plates = QListWidget(self)

        layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.setSpacing(5)  # 간격 조절
        left_layout.addWidget(QLabel("실시간 영상"))
        left_layout.addWidget(self.label_live)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(5)  # 간격 조절
        right_layout.addWidget(QLabel("분석된 이미지"))
        right_layout.addWidget(self.label_captured)
        right_layout.addWidget(QLabel("인식된 번호판"))
        right_layout.addWidget(self.list_plates)


        layout.addLayout(left_layout)
        layout.addLayout(right_layout)
        self.setLayout(layout)

    def showEvent(self, event):
        """ 창이 나타날 때 키보드 포커스를 강제로 잡음 """
        self.grabKeyboard()
        super().showEvent(event)

    def update_image(self, frame):
        self.display_image(frame, self.label_live)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_T:
            print("T key pressed")  # 🔹 디버깅용 출력
            if self.thread.latest_frame is not None:
                self.captured_image = self.thread.latest_frame.copy()
                plates, processed_img = self.detect_license_plates(self.captured_image)
                self.display_image(processed_img, self.label_captured)
                self.display_plates(plates)

    def detect_license_plates(self, img):
        results = self.model(img)
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.4 and int(cls) == 2:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        plates = []
        for bbox, text, conf in self.reader.readtext(img):
            if conf > 0.3:
                filtered_text = re.sub(r'[^가-힣0-9]', '', text)
                if filtered_text:
                    plates.append(filtered_text)
                    cv2.rectangle(img, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 255), 3)

        return plates, img

    def display_image(self, img, label):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def display_plates(self, plates):
        self.list_plates.clear()
        for idx, plate in enumerate(plates):
            self.list_plates.addItem(f'차량 {idx+1}: {plate}')

    def closeEvent(self, event):
        self.cap.release()
        self.thread.quit()
        event.accept()

class VideoCaptureThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        self.latest_frame = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.change_pixmap_signal.emit(frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
