import cv2
import numpy as np
import matplotlib.pyplot as plt

# 尝试读取图像，并检查它们是否已正确加载
image1 = cv2.imread('D:/opencv/lena.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('D:/opencv/labs/lab03/03.png', cv2.IMREAD_GRAYSCALE)

if image1 is None:
    print("Error loading 'lena.png'. Please check the file path.")
if image2 is None:
    print("Error loading '03.png'. Please check the file path.")

# 如果图像正确加载，则继续处理
if image1 is not None and image2 is not None:
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 使用SIFT找到关键点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # 创建蛮力匹配器对象
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行匹配
    matches = bf.match(descriptors1, descriptors2)

    # 根据距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制前N个匹配项
    N = 50  # 可以调整匹配显示的数量
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示图像
    plt.figure(figsize=(12, 6))
    plt.imshow(matched_image, 'gray')
    plt.show()
