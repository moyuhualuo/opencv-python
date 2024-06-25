import cv2
import numpy as np
import os
import csv

def extract_hsv_features(image_path):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    cv2.normalize(h_hist, h_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(s_hist, s_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(v_hist, v_hist, 0, 255, cv2.NORM_MINMAX)
    h_hist = h_hist.flatten()
    s_hist = s_hist.flatten()
    v_hist = v_hist.flatten()
    feature_vector = np.concatenate((h_hist, s_hist, v_hist))
    return feature_vector

def extract_contour_features(image_path):
    image = cv2.imread(image_path, 0)
    kernel_size = (5, 5)
    sigmaX = 0
    gaussian_blurred = cv2.GaussianBlur(image, kernel_size, sigmaX)
    ret, thresh_img = cv2.threshold(gaussian_blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        hu_moments = cv2.HuMoments(M).flatten()
        contour_features.extend([area, perimeter, cX, cY])
        contour_features.extend(hu_moments)
        break  # 如果有多个轮廓，只取第一个
    return contour_features

image_dir = r'D:\opencv\hws\hw04\archive\classified_train\orange'
save_path = (r'D:\opencv\hws\hw04\特征\orange\features.csv')

image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
features_list = []

for image_file in image_files:
    img_path = os.path.join(image_dir, image_file)
    hsv_features = extract_hsv_features(img_path)
    contour_features = extract_contour_features(img_path)
    features = np.concatenate((hsv_features, contour_features))
    features_list.append(features)

with open(save_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for features in features_list:
        writer.writerow(features)

print(f"Saved features to {save_path}")
