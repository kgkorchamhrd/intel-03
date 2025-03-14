import cv2                           # OpenCV 라이브러리 불러오기
import mediapipe as mp               # MediaPipe 라이브러리 불러오기 (mp로 축약)
import random                        # 컴퓨터의 랜덤 선택을 위해 random 모듈 불러오기
import numpy as np                   # 배열 처리를 위한 numpy 불러오기

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands        # MediaPipe의 hands 솔루션 할당
mp_drawing = mp.solutions.drawing_utils  # 손 랜드마크 그리기를 위한 유틸리티 불러오기
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# Hands 객체 초기화 (최소 인식 및 추적 신뢰도 0.5 설정)

# 손가락 개수로 가위, 바위, 보 판별하는 함수 정의
def get_hand_gesture(hand_landmarks):  # 손 랜드마크 정보를 받아 제스처 문자열 반환 함수 정의
    finger_tips = [8, 12, 16, 20]       # 검지, 중지, 약지, 새끼손가락 끝 번호 리스트
    thumb_tip = 4                      # 엄지손가락 끝 번호

    fingers = []                       # 각 손가락의 상태(펴짐 여부)를 저장할 리스트 초기화
    for tip in finger_tips:            # 각 손가락 끝 번호에 대해 반복
        # 손가락 끝 부분이 해당 손가락의 두번째 마디보다 위에 있으면 손가락이 펴진 것으로 판단
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

    # 엄지손가락은 오른손 기준으로 판단 (왼손이면 반대로 판단 필요)
    thumb = hand_landmarks.landmark[thumb_tip].x > hand_landmarks.landmark[thumb_tip - 2].x

    count = fingers.count(True)        # 펴진 손가락의 개수 계산

    if count == 0:
        return "Rock"                  # 모든 손가락이 접힌 경우: 바위
    elif count == 2 and fingers[0] and fingers[1]:
        return "Scissors"              # 검지와 중지만 펴진 경우: 가위
    elif count == 4:
        return "Paper"                 # 네 손가락 모두 펴진 경우: 보

    return "Unknown"                   # 위 조건에 해당하지 않으면 인식 불가로 반환

# 컴퓨터의 제스처 이미지를 미리 로드 (동일 폴더에 rock.png, paper.png, scissor.png 파일 필요)
img_rock = cv2.imread('rock.png')         # 바위 이미지 로드
img_paper = cv2.imread('paper.png')       # 보 이미지 로드
img_scissors = cv2.imread('scissor.png')   # 가위 이미지 로드

# 웹캠 실행 및 환경 설정
cap = cv2.VideoCapture(0)            # 기본 웹캠(0번) 캡처 객체 생성
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 캡처 프레임 너비를 1280으로 설정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 캡처 프레임 높이를 720으로 설정
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)    # 밝기를 150으로 설정 (0~255 범위)
cap.set(cv2.CAP_PROP_CONTRAST, 150)      # 대비를 150으로 설정 (0~255 범위)
cap.set(cv2.CAP_PROP_EXPOSURE, 0)        # 노출 값을 0으로 설정

while cap.isOpened():                # 웹캠이 열려있는 동안 반복 실행
    ret, frame = cap.read()          # 웹캠으로부터 프레임 읽기
    if not ret:                      # 프레임 읽기에 실패하면
        continue                     # 다음 루프로 넘어감

    frame = cv2.flip(frame, 1)       # 프레임을 좌우 반전 (미러 효과)
    cv2.imshow('RPS Game', frame)    # 현재 웹캠 프레임을 윈도우에 표시

    key = cv2.waitKey(1) & 0xFF      # 키 입력을 1밀리초 기다림
    if key == ord('q'):              # 만약 'q' 키를 누르면
        break                      # 반복문 종료하여 프로그램 종료

    # s키를 누르면 게임 라운드 1번 진행
    if key == ord('s'):
        # 현재 프레임을 복사하여 게임 진행용 프레임으로 사용
        game_frame = frame.copy()
        # 복사한 프레임을 RGB로 변환 (MediaPipe는 RGB 이미지를 사용)
        frame_rgb = cv2.cvtColor(game_frame, cv2.COLOR_BGR2RGB)
        # MediaPipe로 손 인식 처리
        results = hands.process(frame_rgb)

        # 인식된 손이 있다면
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:  # 인식된 각 손에 대해 반복
                # 손의 랜드마크와 연결선을 game_frame에 그림
                mp_drawing.draw_landmarks(game_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손 제스처 판별 함수 호출하여 플레이어의 가위바위보 선택 저장
                hand_gesture = get_hand_gesture(hand_landmarks)
                # 컴퓨터의 가위바위보 선택을 무작위로 결정
                computer_choice = random.choice(["Rock", "Paper", "Scissors"])

                # 승패 결정 로직
                if hand_gesture == computer_choice:
                    result_text = "Draw"     # 둘 다 같으면 무승부
                elif (hand_gesture == "Rock" and computer_choice == "Scissors") or \
                     (hand_gesture == "Scissors" and computer_choice == "Paper") or \
                     (hand_gesture == "Paper" and computer_choice == "Rock"):
                    result_text = "Player Wins"  # 플레이어 승리 조건
                else:
                    result_text = "Computer Wins"  # 그 외의 경우 컴퓨터 승리

                # 플레이어, 컴퓨터 선택 및 결과 텍스트를 game_frame에 표시
                cv2.putText(game_frame, f"Player: {hand_gesture}", (100, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)  # 초록색 텍스트
                cv2.putText(game_frame, f"Computer", (920, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)  # 파란색 텍스트
                cv2.putText(game_frame, f"Result: {result_text}", (300, 600),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)  # 빨간색 텍스트

                # 컴퓨터의 제스처에 맞는 이미지를 선택
                if computer_choice == "Rock":
                    comp_img = img_rock
                elif computer_choice == "Paper":
                    comp_img = img_paper
                elif computer_choice == "Scissors":
                    comp_img = img_scissors
                else:
                    comp_img = None

                if comp_img is not None:
                    # 컴퓨터 제스처 이미지를 원하는 크기로 리사이즈 (예: 200x200)
                    comp_img = cv2.resize(comp_img, (200, 200))
                    # 이미지를 표시할 위치를 설정 (예: 우측 상단, 좌표: x=프레임 너비-220, y=20)
                    x_offset = game_frame.shape[1] - 300
                    y_offset = 120
                    # 컴퓨터 제스처 이미지를 game_frame의 해당 영역에 덮어쓰기
                    game_frame[y_offset:y_offset+200, x_offset:x_offset+200] = comp_img
        else:
            # 손 인식이 안 될 경우 텍스트 표시
            cv2.putText(game_frame, "No hand detected", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 게임 라운드 결과가 표시된 프레임을 윈도우에 출력하고
        cv2.imshow('RPS Game', game_frame)
        # 사용자가 아무 키나 누를 때까지 대기 (결과 확인 후 계속 진행)
        cv2.waitKey(0)

cap.release()                        # 웹캠 자원 해제
cv2.destroyAllWindows()              # 모든 OpenCV 창 닫기
