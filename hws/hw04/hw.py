import cv2
import os
import numpy as np

files = r'D:\opencv\hws\hw04\archive\classified_train'
img_files = []
img_dict = {}

for file in os.walk(files):
    if file[0] != files:
        img_files.append(file[0])
#print(img_files, '')


for file in img_files:
    for fl in os.walk(file):
        #print(fl[2])
        pre = fl[2][0].split('_')
        if pre[0] not in img_dict:
            img_dict[pre[0]] = []
        for i in fl[2]:
            img_dict[pre[0]].append(i)
#print(img_dict)

for key, val in img_dict.items():
    print(key, val)

img_path = r'D:\opencv\hws\hw04\archive\classified_train\banana\banana_10.jpg'
img_o = cv2.imread(img_path, 0)
img = cv2.resize(img_o, (240, 240))
cv2.imshow('2', img)
ret, thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('1', thresh_img)


t = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 43, 2)
cv2.imshow('3', t)
cv2.waitKey()
cv2.destroyAllWindows()