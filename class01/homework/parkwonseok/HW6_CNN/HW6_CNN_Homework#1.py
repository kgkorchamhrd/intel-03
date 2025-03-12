import cv2
import numpy as np
img = cv2.imread('/home/park/workspace/intel-03/class01/homework/parkwonseok/HW6_CNN/lena.png', cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1],[1, -8, 1], [1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/edge.png', output)


#identity
kernel = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('identity', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/dentity.png', output)

#Ridge or edge detection
kernel = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Ridge or edge detection', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/Ridge or edge detection1.png', output)

kernel = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Ridge or edge detection', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/Ridge or edge detection2.png', output)

#sharpen
kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('sharpen', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/sharpen.png', output)

#box blur
kernel = (1/9) * np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('box blur', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/box blur.png', output)

#Gaussian blur 3 x 3
kernel = (1/16) * np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Gaussian blur 3 x 3', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/Gaussian blur 3 x 3.png', output)

#Gaussian blur 5 x 5
kernel = (1/256) * np.array([[1,  4,  6,  4,  1],[4, 16, 24, 16,  4],[6, 24, 36, 24,  6],
                             [4, 16, 24, 16,  4],[1,  4,  6,  4,  1]])
print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('Gaussian blur 5 x 5', output)
cv2.waitKey(0)
cv2.imwrite('./blurs/Gaussian blur 5 x 5.png', output)