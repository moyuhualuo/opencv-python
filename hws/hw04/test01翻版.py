import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_and_preprocess_image(file_path, target_size=(100, 100)):
    image = cv2.imread(file_path)
    if image is None:
        print(f"无法加载图像: {file_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    image = cv2.resize(image, target_size)  # 调整大小为目标尺寸
    image = image.astype(np.uint8)  # 确保数据类型为 uint8

    return image


def load_images_from_directory(directory, num_images=None):
    images = []
    file_names = []

    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image = load_and_preprocess_image(file_path)
                if image is not None:
                    images.append(image)
                    file_names.append(file)
                    count += 1
                    if num_images is not None and count >= num_images:
                        return images, file_names

    return images, file_names


def extract_sift_features(images):
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []

    for image in images:
        keypoints, descriptors = sift.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors if descriptors is not None else np.array([]))

    return keypoints_list, descriptors_list


# 设置训练集图片所在的文件夹路径和每个类别的图片数量
train_directory = r'D:\opencv\hws\hw04\archive\classified_train'
num_apple_images = 54
num_banana_images = 63
num_orange_images = 59
num_mixed_images = 0

# 加载苹果图片
apple_images, apple_file_names = load_images_from_directory(os.path.join(train_directory, 'apple'), num_apple_images)

# 加载香蕉图片
banana_images, banana_file_names = load_images_from_directory(os.path.join(train_directory, 'banana'),
                                                              num_banana_images)

# 加载橙子图片
orange_images, orange_file_names = load_images_from_directory(os.path.join(train_directory, 'orange'),
                                                              num_orange_images)

# 加载混合图片
mixed_images, mixed_file_names = load_images_from_directory(os.path.join(train_directory, 'mixed'), num_mixed_images)

# 合并所有图像和文件名
train_images = apple_images + banana_images + orange_images + mixed_images
train_file_names = apple_file_names + banana_file_names + orange_file_names + mixed_file_names

# 提取 SIFT 特征
train_keypoints_list, train_descriptors_list = extract_sift_features(train_images)

# 显示提取到特征的图片数量
print(f"共提取到了 {len(train_images)} 张图片的 SIFT 特征")

# 定义标签
train_labels = ['apple'] * num_apple_images + ['banana'] * num_banana_images + ['orange'] * num_orange_images + [
    'mixed'] * num_mixed_images
train_descriptors_array = np.vstack([desc for desc in train_descriptors_list if desc.size > 0])

# 设置视觉词汇的数量（聚类中心数）
num_clusters = 300  # 增加到 300

# 使用K-Means聚类对所有特征描述符进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(train_descriptors_array)

# 获取聚类中心（视觉词汇）
visual_words = kmeans.cluster_centers_


def generate_bow_histogram(descriptors, visual_words):
    histogram = np.zeros(len(visual_words))
    if descriptors is not None and len(descriptors) > 0:
        for descriptor in descriptors:
            distances = np.linalg.norm(visual_words - descriptor, axis=1)
            nearest_visual_word_index = np.argmin(distances)
            histogram[nearest_visual_word_index] += 1
    return histogram


# 生成所有图片的Bag of Words直方图，并排除无效直方图的图片和标签
filtered_train_histograms = []
filtered_train_labels = []

for descriptors, label in zip(train_descriptors_list, train_labels):
    if descriptors.size > 0:
        histogram = generate_bow_histogram(descriptors, visual_words)
        if histogram.sum() > 0:  # 排除直方图全零的情况
            filtered_train_histograms.append(histogram)
            filtered_train_labels.append(label)

train_histograms = np.array(filtered_train_histograms)
train_labels = np.array(filtered_train_labels)

print(f"Number of histograms: {len(train_histograms)}")
print(f"Number of labels: {len(train_labels)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_histograms, train_labels, test_size=0.2, random_state=42)

# 构建 SVM 管道
svm = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42))

# 定义参数网格
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [1e-3, 1e-4, 'scale', 'auto']
}

# 使用网格搜索找到最佳参数
grid_search = GridSearchCV(svm, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print(f"最佳参数: {grid_search.best_params_}")

# 使用最佳参数的模型
best_svm = grid_search.best_estimator_

# 评估最佳模型
accuracy = best_svm.score(X_test, y_test)
print(f"分类器在测试集上的准确率: {accuracy:.2f}")


def predict_image_category(image_path, visual_words, classifier):
    # 预处理图像
    image = load_and_preprocess_image(image_path)
    if image is None:
        return None

    # 提取SIFT特征
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # 生成Bag of Words直方图
    histogram = generate_bow_histogram(descriptors, visual_words)

    # 预测类别
    prediction = classifier.predict([histogram])
    return prediction[0]


# 预测新图像的类别
image_path = r'D:\opencv\hws\hw04\archive\classified_train\apple\apple_1.jpg'
predicted_category = predict_image_category(image_path, visual_words, best_svm)
print(f"预测类别: {predicted_category}")
"""
第二次调整开始！
"""



def augment_image(image):
    # 可以添加更多的增强方法
    augmented_images = [image]
    augmented_images.append(cv2.flip(image, 1))  # 水平翻转
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))  # 顺时针旋转90度
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE))  # 逆时针旋转90度
    return augmented_images

def load_images_from_directory_with_augmentation(directory, num_images=None):
    images = []
    file_names = []

    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image = load_and_preprocess_image(file_path)
                if image is not None:
                    augmented_images = augment_image(image)
                    for aug_image in augmented_images:
                        images.append(aug_image)
                        file_names.append(file)
                    count += len(augmented_images)
                    if num_images is not None and count >= num_images:
                        return images, file_names

    return images, file_names

# 加载增强后的苹果图片
apple_images, apple_file_names = load_images_from_directory_with_augmentation(os.path.join(train_directory, 'apple'))

# 加载增强后的香蕉图片
banana_images, banana_file_names = load_images_from_directory_with_augmentation(os.path.join(train_directory, 'banana'))

# 加载增强后的橙子图片
orange_images, orange_file_names = load_images_from_directory_with_augmentation(os.path.join(train_directory, 'orange'))

# 合并所有图像和文件名
train_images = apple_images + banana_images + orange_images
train_file_names = apple_file_names + banana_file_names + orange_file_names
# 设置视觉词汇的数量（聚类中心数）
num_clusters = 500  # 增加到 500

# 使用K-Means聚类对所有特征描述符进行聚类
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(train_descriptors_array)

# 获取聚类中心（视觉词汇）
visual_words = kmeans.cluster_centers_
# 尝试使用随机森林分类器
from sklearn.ensemble import RandomForestClassifier

rf = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))

# 定义参数网格
param_grid_rf = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 10, 20, 30]
}

# 使用网格搜索找到最佳参数
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=3)
grid_search_rf.fit(X_train, y_train)

# 输出最佳参数
print(f"随机森林最佳参数: {grid_search_rf.best_params_}")

# 使用最佳参数的模型
best_rf = grid_search_rf.best_estimator_

# 评估最佳模型
accuracy_rf = best_rf.score(X_test, y_test)
print(f"随机森林分类器在测试集上的准确率: {accuracy_rf:.2f}")
"""
第三次调整开始


"""

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 定义提取颜色直方图的函数
def extract_color_histogram(image, bins=(8, 8, 8)):
    # 将图像从BGR转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 计算颜色直方图
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # 归一化直方图
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# 加载和预处理图像
def load_images_from_directory(directory):
    images = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                image = cv2.imread(file_path)
                if image is not None:
                    hist = extract_color_histogram(image)
                    images.append(hist)
                    # 根据文件夹名称添加标签
                    label = os.path.basename(root)
                    labels.append(label)
                else:
                    print(f"无法加载图像: {file_path}")
    return np.array(images), np.array(labels)

# 定义数据路径
train_directory = r'D:\opencv\hws\hw04\archive\classified_train'

# 加载数据
X, y = load_images_from_directory(train_directory)

# 检查是否成功加载数据
print(f"加载到的图像数量: {len(X)}")
print(f"加载到的标签数量: {len(y)}")

# 如果数据为空，则输出提示并退出
if len(X) == 0 or len(y) == 0:
    print("没有加载到任何图像，请检查数据路径和图像文件格式。")
else:
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用SVM分类器
    svm = SVC()

    # 定义参数网格
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf']
    }

    # 使用网格搜索找到最佳参数
    grid_search_svm = GridSearchCV(svm, param_grid_svm, cv=3)
    grid_search_svm.fit(X_train, y_train)

    # 输出最佳参数
    print(f"SVM最佳参数: {grid_search_svm.best_params_}")

    # 使用最佳参数的模型
    best_svm = grid_search_svm.best_estimator_

    # 评估最佳模型
    y_pred = best_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred)
    print(f"SVM分类器在测试集上的准确率: {accuracy_svm:.2f}")

    # 显示结果
    for img_path, pred, actual in zip(X_test, y_pred, y_test):
        print(f"预测: {pred}, 实际: {actual}")

