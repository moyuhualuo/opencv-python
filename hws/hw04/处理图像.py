import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r'D:\opencv\hws\hw04\archive\classified_train\orange\orange_1.jpg'
img_o = cv2.imread(img_path, 0) # 读取为灰度图
img = cv2.resize(img_o, (240, 240))
# 应用高斯滤波去噪
kernel_size = (5, 5)  # 高斯核大小，必须是奇数
sigmaX = 0  # 如果 sigmaX 为 0，那么它会根据核函数大小来计算
gaussian_blurred = cv2.GaussianBlur(img, kernel_size, sigmaX)
# 二值化
ret, thresh_img = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#cv2.imshow('Thresh Image', thresh_img)

t = cv2.adaptiveThreshold(gaussian_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43, 2)
cv2.imshow('3', t)

# 显示原始图像、高斯滤波后的图像和二值化后的图像
cv2.imshow('Original Image', img)
cv2.imshow('Gaussian Blurred Image', gaussian_blurred)
cv2.imshow('Thresholded Image', thresh_img)

# 等待按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


