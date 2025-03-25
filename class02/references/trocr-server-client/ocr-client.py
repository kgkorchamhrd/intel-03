# client.py
import requests
from time import time

# 서버 및 인증 설정
PORT_N = 5050
SERVER_IP = "61.108.166.15"
USER_ID = "team01"
USER_PW = "1234"
TEST_IMG = "test2.png"

# 요청 URL 생성
url = f"http://{SERVER_IP}:{PORT_N}/ocr"
auth = (USER_ID, USER_PW)

# 이미지 파일을 binary 모드로 열어 요청 데이터 준비
with open(TEST_IMG, "rb") as img_file:
    files = {"image": img_file}

    # 요청 시작 시간 측정
    start_time = time()

    # 서버에 POST 요청 전송 (이미지 업로드 및 인증 정보 포함)
    response = requests.post(url, auth=auth, files=files)

    # 요청 종료 시간 측정
    end_time = time()
    print(f"처리 시간: {end_time - start_time:.2f}초")

# 응답 상태 코드에 따른 결과 출력
if response.status_code == 200:
    print("OCR Text:", response.json()["text"])
else:
    print("Error:", response.status_code, response.json())
