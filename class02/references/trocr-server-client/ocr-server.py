"""
Flask 기반 OCR 서버 코드
- 클라이언트가 전송한 이미지를 OCR 처리 후 텍스트를 반환합니다.
- 사용한 AI model: https://huggingface.co/ddobokki/ko-trocr
"""

import unicodedata
from time import time

import psutil
import torch

from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

# 서버 및 인증 설정 상수
PORT_N = 5050
SERVER_IP = "61.108.166.15"
USER_ID = "team01"
USER_PW = "1234"


def free_port(port=PORT_N):
    """
    지정된 포트를 사용 중인 프로세스를 종료하여 포트를 해제하는 함수.
    :param port: 해제할 포트 번호 (기본값: PORT_N)
    """
    for proc in psutil.process_iter(['pid']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr and conn.laddr.port == port:
                    proc.kill()
                    print(f"프로세스 {proc.pid} 종료: 포트 {port} 사용 중! 재실행 요망!")
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue


# Flask 앱 초기화
app = Flask(__name__)

# 모델, 프로세서, 토크나이저 로드 및 GPU 설정
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def authenticate(auth):
    """
    HTTP Basic 인증 정보를 검증하는 함수.
    :param auth: 요청에 포함된 인증 정보
    :return: 인증 성공 여부 (True/False)
    """
    return auth and auth.username == USER_ID and auth.password == USER_PW


@app.route('/ocr', methods=['POST'])
def ocr():
    """
    /ocr 엔드포인트:
    - 클라이언트가 전송한 이미지를 OCR 처리하여 텍스트를 반환합니다.
    - 처리 시간도 출력합니다.
    """
    start_time = time()

    # 인증 검사
    auth = request.authorization
    if not authenticate(auth):
        return jsonify({"error": "Unauthorized"}), 401

    # 이미지 파일이 요청에 포함되어 있는지 확인
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    try:
        # 이미지를 RGB 모드로 변환하여 로드 (필수)
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400

    # 이미지 전처리 및 모델 추론
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, max_length=64)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    generated_text = unicodedata.normalize("NFC", generated_text)

    end_time = time()
    print(f"처리 시간: {end_time - start_time:.2f}초")
    return jsonify({"text": generated_text})


if __name__ == '__main__':
    # 서버 시작 전 지정된 포트를 해제
    free_port(PORT_N)
    app.run(host=SERVER_IP, port=PORT_N, debug=False, use_reloader=False)

