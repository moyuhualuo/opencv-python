import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread(R'D:\opencv\lena.png', cv2.IMREAD_GRAYSCALE)

# 定义结构元素
kernel = np.ones((5,5), np.uint8)

# 顶帽运算
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

# 黑帽运算
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

# 显示图像
plt.figure(figsize=(10, 5))
titles = ['Original Image', 'Top Hat', 'Black Hat']
images = [image, tophat, blackhat]

for i in range(3):
    plt.subplot(1, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
