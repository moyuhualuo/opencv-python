import cv2
import numpy as np


def make():
    src = np.zeros((8, 8), dtype=np.uint8)
    print('src-0:\n', src)
    '''
    >>> What's the picture of zero?
    0 means Black, 255 means White.
    '''
    cv2.imshow('one', src)

    print('src[0][2]=', src[0][2])
    src[0][2] = 255
    print('src-1:\n', src)
    print(src[0][2])
    cv2.imshow('two', src)
    '''
    >>> 'what dif one and two?'
    '''
    cv2.waitKey()
    cv2.destroyAllWindows()


def alter():
    """
    You can code it you like.
    This is just a case.
    0 means Black, 255 means White.
    """
    img = cv2.imread(r"D:\opencv\lab01\brush-strokes-6139429_1280.png", 0)
    cv2.imshow('before', img)
    '''
    >>> We alter it, it will be white.
    '''
    for i in range(90, 401):
        for j in range(160, 521):
            img[i, j] = 255
    cv2.imshow('after', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    '''
    What dif we see.
    '''


def bgr():
    """
    BGR
    """
    blue = np.zeros((300, 300, 3), dtype=np.uint8)
    blue[:, :, 0] = 255  # ':' means all
    cv2.imshow('Blue', blue)

    green = np.zeros((300, 300, 3), dtype=np.uint8)
    green[:, :, 1] = 255  # ':' means all
    cv2.imshow('Green', green)

    red = np.zeros((300, 300, 3), dtype=np.uint8)
    red[:, :, 2] = 255  # ':' means all
    cv2.imshow('Red', red)

    cv2.waitKey()
    cv2.destroyAllWindows()


def brg_one():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, 0:100, 0] = 255
    img[:, 100:200, 1] = 255
    img[:, 200:300, 2] = 255

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


"""
Please del '#' to run function!
"""

#  make()
#  alter()
#  bgr() #  What do you see?
#  brg_one()


'''
>>> Please use your knowledge code the picture of char 'a'. 
'''

