'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置图像目录路径
image_dir = r'D:\opencv\hws\hw04\archive\classified_train\orange'
# 设置保存直方图的目录路径
save_dir = r'D:\opencv\hws\hw04\特征\orange'

# 如果保存目录不存在，则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取目录中的所有文件
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# 遍历所有图像文件
for i, image_file in enumerate(image_files):
    # 读取图像
    img_path = os.path.join(image_dir, image_file)
    img = cv2.imread(img_path)

    # 检查图像是否成功加载
    if img is None:
        print(f"Failed to load image {img_path}")
        continue

    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 计算三个通道的直方图
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # 归一化直方图
    cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)

    # 将直方图数据转换为8位无符号整数
    h_hist = np.int32(np.around(h_hist))
    s_hist = np.int32(np.around(s_hist))
    v_hist = np.int32(np.around(v_hist))

    # 由于matplotlib没有'h'、's'、'v'颜色，我们需要使用RGB或其他颜色代替
    # 这里我们使用一些代表性的颜色
    h_color = 'blue'  # H通道通常用蓝色表示
    s_color = 'green'  # S通道用绿色表示饱和度
    v_color = 'red'  # V通道用红色表示亮度

    plt.figure()
    plt.plot(h_hist, color=h_color, label='Hue')
    plt.plot(s_hist, color=s_color, label='Saturation')
    plt.plot(v_hist, color=v_color, label='Value')

    plt.title(f'Color Histograms for HSV Channels - {image_file}')
    plt.xlabel('Bins')
    plt.ylabel('Counts')
    plt.xlim([0, 256])  # 对于S和V通道，尽管只有180个bin，但为了统一显示，我们设置xlim为256
    plt.legend()

    # 保存图表到文件
    save_path = os.path.join(save_dir, f'histogram_{i}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"Saved histogram for {image_file} to {save_path}")
'''
import cv2
import numpy as np
import os


def extract_hsv_features(image_path):
    # 读取图像
    img = cv2.imread(image_path)

    # 将图像从BGR转换为HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 计算三个通道的直方图
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # 归一化直方图
    cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)

    # 将直方图数据转换为一维数组
    h_hist = h_hist.flatten()
    s_hist = s_hist.flatten()
    v_hist = v_hist.flatten()

    # 拼接成一个特征向量
    feature_vector = np.concatenate((h_hist, s_hist, v_hist))

    return feature_vector


# 设置图像目录路径
image_dir = r'D:\opencv\hws\hw04\archive\classified_train\orange'
# 设置保存特征的目录路径
feature_dir = r'D:\opencv\hws\hw04\特征\orange'

# 如果保存目录不存在，则创建
if not os.path.exists(feature_dir):
    os.makedirs(feature_dir)

# 获取目录中的所有文件
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# 初始化一个字典来保存所有特征
features_dict = {}

# 遍历所有图像文件并提取特征
for i, image_file in enumerate(image_files):
    img_path = os.path.join(image_dir, image_file)
    feature_vector = extract_hsv_features(img_path)
    features_dict[image_file] = feature_vector
    print(f"Extracted features for {image_file}")

# 保存特征到文件
features_path = os.path.join(feature_dir, 'hsv_features.npy')
np.save(features_path, features_dict)
print(f"Saved features to {features_path}")
