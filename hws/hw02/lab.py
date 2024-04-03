import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def function_01(): # 调用cv2.cvtColor(imgName,cv2.COLOR_RGB2GRAY)函数可以将彩色图像转换为灰度图像。
    img = cv2.imread('track_plate.png', -1)
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(img)


def function_02(): # 三通道图形
    img = cv2.imread('track_plate.png')
    B, G, R = cv2.split(img)
    print(B.shape, G.shape, R.shape)
    print(img.shape)
    print(B)
    res = cv2.hconcat([B, G, R])
    cv2.imshow('res', res)
    cv2.waitKey()
    cv2.destroyAllWindows()
    hist_color00 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_color01 = cv2.calcHist([img], [1], None, [256], [0, 256])
    hist_color02 = cv2.calcHist([img], [2], None, [256], [0, 256])
    plt.figure()

    plt.plot(hist_color00, c='b', label='B')
    plt.plot(hist_color01, c='g', label='G')
    plt.plot(hist_color02, c='r', label='R')
    # 绘制 B 通道直方图
    plt.hist(hist_color00, bins=40, color='blue', label='B', alpha=0.5)
    # 绘制 G 通道直方图
    plt.hist(hist_color01, bins=40, color='green', label='G', alpha=0.3)
    # 绘制 R 通道直方图
    plt.hist(hist_color02, bins=40, color='red', label='R', alpha=0.4)
    plt.title('Colors Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def function_03():  # 灰度直方图
    img = cv2.imread('track_plate.png', -1)
    hist_gary = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist_gary)
    plt.title('Gray Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()
    plt.close('all')


def function_04():  # 计算二维高斯核
    kernel_size = 5
    sigma = 0.7
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

    img = cv2.imread('track_plate.png', -1)
    dst = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    smoothed_img = cv2.filter2D(dst, -1, gaussian_kernel_2d)
    res = cv2.hconcat([dst, smoothed_img])
    cv2.imshow('res', res)
    cv2.waitKey()
    cv2.destroyAllWindows()

    hist_dst = cv2.calcHist([dst], [0], None, [256], [0, 256])
    hist_smoothed_img = cv2.calcHist([smoothed_img], [0], None, [256], [0, 256])

    plt.figure()
    sns.histplot(hist_dst, label='origin',  palette="cool")
    sns.histplot(hist_smoothed_img, label='smoothed',  palette="hot")
    plt.legend()
    plt.show()
    plt.close('all')

function_04()
def function_05():
    image = cv2.imread('track_plate.png', cv2.IMREAD_GRAYSCALE)
    # 对图像进行均值化
    equalized_image = cv2.equalizeHist(image)
    # 显示原始图像和均值化后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.show()
    plt.close('all')

    hist_gary = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_dst = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist_gary, label='gary')
    plt.plot(hist_dst, label='dst')
    plt.title('Gray Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    plt.close('all')
