import cv2
import sys
import numpy as np
import pytesseract
from PIL import Image
import re


def find_candidates(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours]  # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
                  for center, size, angle in rects if verify_aspect_size(size)]

    return candidates


def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산

    chk1 = 3000 < (h * w) < 15000          # 번호판 넓이 조건
    chk2 = 2.5 < aspect < 8.0       # 번호판 종횡비 조건

    #print(w,h)
    return (chk1 and chk2)


def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # BGR 컬러 영상을 명암 영상으로 변환하여 저장
    blur = cv2.blur(gray, (5, 5))
    sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, 3)
    _, b_img = cv2.threshold(sobel, 120, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 17), np.uint8)
    morph = cv2.morphologyEx(b_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    se1 = np.ones((5, 9), np.uint8)

    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, se1, iterations=3)

    se2 = np.ones((6, 5), np.uint8)
    morph = cv2.dilate(morph, se2, iterations=5)

    return morph


def getWarpPerspectiveRectImg(candidate):
    src_points = np.float32(cv2.boxPoints(candidate))

    # 왼쪽이 내려간 직사각형
    if src_points[1][0] > src_points[3][0]:
        dst_points = np.float32([(0, 0),
                                 (np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (
                                             src_points[0][1] - src_points[1][1]) ** 2), 0),
                                 (np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (
                                             src_points[0][1] - src_points[1][1]) ** 2), np.sqrt(
                                     (src_points[1][0] - src_points[2][0]) ** 2 + (
                                                 src_points[1][1] - src_points[2][1]) ** 2)),
                                 (0, np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (
                                             src_points[1][1] - src_points[2][1]) ** 2))])

        perspect_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        rotatedRect = cv2.warpPerspective(img, perspect_mat, np.int32((np.sqrt(
            (src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2), np.sqrt(
            (src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2))), cv2.INTER_CUBIC)

    # 오른쪽이 내려간 직사각형
    else:
        dst_points = np.float32(
            [(0, np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2)),
             (0, 0),
             (np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2), 0),
             (np.sqrt((src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2),
              np.sqrt((src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2))])

        perspect_mat = cv2.getPerspectiveTransform(src_points, dst_points)
        rotatedRect = cv2.warpPerspective(img, perspect_mat, np.int32((np.sqrt(
            (src_points[1][0] - src_points[2][0]) ** 2 + (src_points[1][1] - src_points[2][1]) ** 2), np.sqrt(
            (src_points[0][0] - src_points[1][0]) ** 2 + (src_points[0][1] - src_points[1][1]) ** 2))), cv2.INTER_CUBIC)

    return rotatedRect


car_no = str(input("자동차 영상 번호 (00~09): "))
img = cv2.imread('cars/'+car_no+'.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')
cv2.imshow('original',img)


# 1 전처리 단계 (hw2-2)
preprocessed = preprocessing(img)

cv2.imshow('plate candidate 0', preprocessed)


# 2 번호판 후보 영역 검출 (hw3-2)
candidates = find_candidates(preprocessed)

img2 = img.copy()
for candidate in candidates:  # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(img2, [pts], True, (0, 225, 255), 3)

cv2.imshow('plate candidate 1', img2)


# 3 각 후보마다 warp 변환으로 회전하여 번호판 인식
candidateNum = 0
for candidate in candidates:
    rotatedRect = getWarpPerspectiveRectImg(candidate)

    grayRotatedRect = cv2.cvtColor(rotatedRect, cv2.COLOR_BGR2GRAY)

    bilateralImg = cv2.bilateralFilter(grayRotatedRect, 11, 17, 17)

    multi_license_plate = cv2.multiply(bilateralImg, 1.64)

    (thresh, TOZERO_license_plate) = cv2.threshold(multi_license_plate, 180, 255, cv2.THRESH_TOZERO)
    cv2.imshow('TOZERO_license_plate' + str(candidateNum), TOZERO_license_plate)

    candidateNum += 1

    img_pil = Image.fromarray(TOZERO_license_plate)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    result = pytesseract.image_to_string(img_pil, lang='kor')
    result = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", result)
    result = result.replace(" ", "")
    if result:
        print(result)

cv2.waitKey()
cv2.destroyAllWindows()