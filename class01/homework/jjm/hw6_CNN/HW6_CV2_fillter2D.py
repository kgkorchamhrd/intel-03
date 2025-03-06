
import numpy as np
import cv2

img = cv2.imread("./src_img/image.jpg", cv2.IMREAD_COLOR)
# kernel = np.array([[0, 0, 0],[0, 1, 0], [0, 0, 0]])                     # idenity
# kernel = np.array([[0, -1, 0],[-1, 4, -1], [0, -1, 0]])                 # Ridge
# kernel = np.array([[-1, -1, -1],[-1, 8, -1], [-1, -1, -1]])             # edge_detection
# kernel = np.array([[0, -1, 0],[-1, 5, -1], [0, -1, 0]])                 # Sharpen
# kernel = np.array([[1, 1, 1],[1, 1, 1], [1, 1, 1]])                     # Box_blur
# kernel = np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]]) / 16                # Gaussian_blur_3x3
kernel = np.array([[1, 4, 6, 4, 1], 
                   [4, 16, 24, 16, 4], 
                   [6, 24,36, 24, 6], 
                   [4, 16, 24, 16, 4], 
                   [1, 4, 6, 4, 1]]) / 256                              # Gaussian_blur_5x5
print(kernel) 

output = cv2.filter2D(img, -1, kernel)
cv2.imshow("output", output)

key = cv2.waitKey(0)
if key == 0x73:     # ASCII 's'
    cv2.imwrite("./src_img/CV2_exam_07_Gaussian_blur_5x5.png", output)

cv2.destroyAllWindows()


