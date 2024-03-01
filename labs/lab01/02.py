import numpy as np
import cv2
# tools
"""
Like img.shape /img.size /img.dtype/......
"""

def random_img():
    """Great idea.
    What can we see?
    """
    scr = np.random.randint(0, 256, size=[256, 256], dtype=np.uint8)
    cv2.imshow('random_img', scr)
    cv2.waitKey()
    cv2.destroyAllWindows()


def random_brg():
    color_img = np.random.randint(0, 256, size=[256, 256, 3], dtype=np.uint8)
    cv2.imshow('brg_img', color_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def mosaic():
    """Now mosaic anywhere we can.
    """
    a = cv2.imread(r'D:\opencv\lab01\brush-strokes-6139429_1280.png')
    cv2.imshow('img', a)
    face = np.random.randint(0, 256, (180, 95, 3))
    a[220:400, 255:350] = face
    cv2.imshow('result', a)
    '''What dif 'result' and 'img', mosaic!
    '''
    cv2.waitKey()
    cv2.destroyAllWindows()


def fake_ps():
    """lIKE ps, but these img are not great sample, ROI"""
    one = cv2.imread(r'D:\opencv\lab01\brush-strokes-6139429_1280.png')
    two = cv2.imread(r'D:\opencv\lab01\girl.png')
    head = two[:100, :100]
    one[:100, :100] = head
    cv2.imshow('head', head)
    cv2.imshow('one', one)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sp():
    img1 = cv2.imread(r'D:\opencv\lab01\brush-strokes-6139429_1280.png')
    b, g, r = cv2.split(img1)  # split B, G, R
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.waitKey()
    """merge B, R, G code
    bgr = cv2.merge([b, g, r])
    what about bgr, rgb?
    Please CODE HEHE 
    """
    cv2.destroyAllWindows()


img = np.random.randint(10, 100, size=[5, 5], dtype=np.uint8)
print(img.item(3, 2))  # before
img.itemset((3, 2), 255)  # set
print(img.item(3, 2))  # after
'''Use item function can improve efficiency.
'''


# random_img()
# random_brg()
# mosaic()
# fake_ps()
# sp()
'''drop '#' to use these functions.
'''
