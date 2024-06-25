import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(R'D:\opencv\lena.png', cv2.IMREAD_GRAYSCALE)

# 定义结构元素
kernel = np.ones((5,5), np.uint8)

# 腐蚀
erosion = cv2.erode(image, kernel, iterations = 1)

# 膨胀
dilation = cv2.dilate(image, kernel, iterations = 1)

# 开运算
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 闭运算
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 形态学梯度
gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# 显示图像
plt.figure(figsize=(10, 8))
titles = ['Original Image', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']
images = [image, erosion, dilation, opening, closing, gradient]

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
