import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time  # 用于测量时间

# 函数：加载CSV文件中的特征
def load_csv_features(csv_file):
    features = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            features.append([float(x) for x in row])
    return np.array(features)

# 函数：加载npy文件中的特征
def load_npy_features(npy_file):
    features_dict = np.load(npy_file, allow_pickle=True).item()
    features = []
    for key in features_dict:
        features.append(features_dict[key])
    return np.array(features)

# 函数：加载所有特征和标签
def load_features(base_dir, label_value):
    csv_file_path = os.path.join(base_dir, 'features.csv')
    npy_file_path = os.path.join(base_dir, 'hsv_features.npy')

    csv_features = load_csv_features(csv_file_path)
    npy_features = load_npy_features(npy_file_path)

    # 合并特征
    features = np.concatenate((csv_features, npy_features), axis=1)

    # 生成标签
    labels = np.full(features.shape[0], label_value)

    return features, labels

# 测量数据加载时间
start_time = time.time()

# 加载所有类别的特征和标签
base_dirs = {
    'apple': r'D:\opencv\hws\hw04\特征\apple',
    'banana': r'D:\opencv\hws\hw04\特征\banana',
    'orange': r'D:\opencv\hws\hw04\特征\orange'
}

all_features = []
all_labels = []

# 对应标签：0 - apple, 1 - banana, 2 - orange
for label_value, (fruit, base_dir) in enumerate(base_dirs.items()):
    features, labels = load_features(base_dir, label_value)
    all_features.append(features)
    all_labels.append(labels)

# 合并所有类别的特征和标签
all_features = np.vstack(all_features)
all_labels = np.concatenate(all_labels)

# 数据加载时间
data_load_time = time.time() - start_time
print(f"Data load time: {data_load_time:.2f} seconds")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# 定义管道，包含PCA降维和SVM分类器
pipe = Pipeline([
    ('pca', PCA(n_components=50)),  # 降低PCA的n_components以加快计算
    ('svm', SVC())
])

# 定义简化的参数网格，用于网格搜索
param_grid = {
    'svm__kernel': ['linear', 'rbf'],
    'svm__C': [0.1, 1],
    'svm__gamma': ['scale']
}

# 使用网格搜索和交叉验证进行超参数优化（减少折数）
start_time = time.time()
grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy')  # 使用3折交叉验证
grid_search.fit(X_train, y_train)
grid_search_time = time.time() - start_time
print(f"Grid search time: {grid_search_time:.2f} seconds")

# 输出最佳参数
print(f"Best parameters: {grid_search.best_params_}")

# 使用最佳模型进行预测
start_time = time.time()
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
prediction_time = time.time() - start_time
print(f"Prediction time: {prediction_time:.2f} seconds")

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 分类报告
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
