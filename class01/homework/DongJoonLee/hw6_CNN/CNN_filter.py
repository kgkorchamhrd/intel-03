import cv2
import numpy as np
img = cv2.imread('image.png', cv2.COLOR_BGR2RGB)
# kernel = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])
# kernel = np.array([[0,-1, 0],[-1, 4, -1], [0, -1, 0]])
# kernel = np.array([[-1, -1, -1],[-1, -8, -1], [-1, -1, -1]])
# kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])
# kernel = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]])
# kernel = kernel*(1/9)
# kernel = np.array([[1, 2, -1],[2, 4, 2], [1, 2, 1]])
# # kernel = kernel*(1/16)
# kernel = np.array([[1, 4, 6,4,1],[4,16,24,16,4], [6, 24,36,24,6], [1, 4, 6,4,1],[4,16,24,16,4]])
# kernel = kernel*(1/256)

print(kernel)
output = cv2.filter2D(img, -1, kernel)
cv2.imshow('edge', output)
cv2.waitKey(0)