import sys
import cv2
import numpy as np
import easyocr
from openvino.runtime import Core
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(5)  # 카메라 ID 설정
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(frame)

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.cap.release()


class LicensePlateRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # OpenVINO 설정
        self.ie = Core()
        self.vehicle_model = self.ie.read_model("vehicle_detection_model/vehicle.xml")
        self.vehicle_compiled_model = self.ie.compile_model(self.vehicle_model, "CPU")
        self.plate_model = self.ie.read_model("yolov5s_openvino_model/yolov5s.xml")
        self.plate_compiled_model = self.ie.compile_model(self.plate_model, "CPU")

        # EasyOCR 설정
        self.reader = easyocr.Reader(['ko', 'en'], gpu=False)

        # 비디오 스레드 시작
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

        # 타이머 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)

        self.latest_frame = None

    def initUI(self):
        self.setWindowTitle('차량 번호판 인식 시스템')
        self.setGeometry(100, 100, 800, 600)
        self.label_video = QLabel(self)
        self.label_video.setFixedSize(640, 480)
        layout = QVBoxLayout()
        layout.addWidget(self.label_video)
        self.setLayout(layout)

    def update_image(self, frame):
        self.latest_frame = frame
        self.display_image(frame)

    def process_frame(self):
        if self.latest_frame is not None:
            self.display_image(self.latest_frame)

    def keyPressEvent(self, event):
        if event.text().lower() == 't' and self.latest_frame is not None:
            cv2.imwrite("captured.jpg", self.latest_frame)
            self.analyze_image("captured.jpg")

    def analyze_image(self, image_path):
        img = cv2.imread(image_path)
        vehicle_boxes = self.detect_objects(img, self.vehicle_compiled_model, threshold=0.5)

        for (vx1, vy1, vx2, vy2) in vehicle_boxes:
            vehicle_roi = img[vy1:vy2, vx1:vx2]
            plate_boxes = self.detect_objects(vehicle_roi, self.plate_compiled_model, threshold=0.4)
            for (px1, py1, px2, py2) in plate_boxes:
                absolute_px1, absolute_py1 = vx1 + px1, vy1 + py1
                absolute_px2, absolute_py2 = vx1 + px2, vy1 + py2
                plate_roi = img[absolute_py1:absolute_py2, absolute_px1:absolute_px2]
                text_results = self.reader.readtext(plate_roi)
                for bbox, text, conf in text_results:
                    if conf > 0.4:
                        filtered_text = ''.join([c for c in text if c.isalnum()])
                        cv2.rectangle(img, (absolute_px1, absolute_py1), (absolute_px2, absolute_py2), (0, 255, 0), 2)
                        cv2.putText(img, filtered_text, (absolute_px1, absolute_py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        self.display_image(img)

    def detect_objects(self, img, model, threshold=0.5):
        input_layer = model.input(0)
        output_layer = model.output(0)
        img_resized = cv2.resize(img, (640, 640))
        img_transposed = img_resized.transpose(2, 0, 1)
        img_input = np.expand_dims(img_transposed, axis=0).astype(np.float32) / 255.0
        results = model([img_input])[output_layer]
        detected_boxes = []
        for det in results[0]:
            conf = det[4]
            if conf > threshold:
                x, y, w, h = det[0], det[1], det[2], det[3]
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                detected_boxes.append((x1, y1, x2, y2))
        return detected_boxes

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = channel * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label_video.setPixmap(pixmap)
        self.label_video.setScaledContents(True)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LicensePlateRecognitionApp()
    window.show()
    sys.exit(app.exec_())
