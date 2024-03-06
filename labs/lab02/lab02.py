import cv2
import numpy as np

"""
# 1.Two pictures but one function, bur shape must same.
# 2.match function
# 3.fun_f function means match_filed, this function like jigsaw puzzles
———————————————————————————————————————————————————————————————————————
‘&’ means and .
>>> cv2.bitwise_and(img1, img2)
‘or’ means or.
>>> cv2.bitwise_or(img1, img2)
'not' means !.
>>> cv2.bitwise_not(img1, img2)
'xor' mans |. Carry-less binary addition
>>> cv2.bitwise_xor(img1, img2)
———————————————————————————————————————————————————————————————————————
cv2.add(img1, 6) same cv2.add(6, img1)
cv2.add(img1, img2)
>>> Increase brightness
"""


def match(address1, address2):
    """
    there no size() or shape() in scr
    """
    one = cv2.imread(address2)
    two = cv2.imread(address1)
    # s1 = one.size
    # s2 = two.size
    sh1 = one.shape
    sh2 = two.shape
    # what's match?
    if sh1 == sh2:
        return True
    else:
        return False


def fun_f(a, b, c, d, address1, address2):
    """If failed, ROI maybe help you"""
    one = cv2.imread(address1)
    two = cv2.imread(address2)
    new_one = one[a:b, c:d]
    new_two = two[a:b, c:d]
    add = cv2.addWeighted(new_one, 0.6, new_two, 0.4, 0)
    one[a:b, c:d] = add
    cv2.imshow('add', add)
    cv2.imshow('one', one)
    cv2.waitKey()
    cv2.destroyAllWindows()


def two_picture(address1, address2):
    one = cv2.imread(address1)
    background = cv2.imread(address2)
    # 0 means gamma, can't ignore
    if match(address1, address2):
        res = cv2.addWeighted(one, 0.6, background, 0.4, 0)
        cv2.imshow('res', res)
        # c = cv2.waitKey()
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('match failed!')


def mask(address1):
    a = cv2.imread(address1)
    b = np.zeros(a.shape, dtype=np.uint8)
    """255 means 11111111"""
    b[100:700, 200:500] = 255
    c = cv2.bitwise_and(a, b)
    cv2.imshow('b', b)
    cv2.imshow('c', c)
    cv2.waitKey()
    cv2.destroyAllWindows()


img = cv2.imread('back.jpg', 0)
cv2.imshow('back', img)
r, c = img.shape
x = np.zeros((r, c, 8), dtype=np.uint8)
for i in range(8):
    x[:, :, i] = 2 ** i
r = np.zeros((r, c, 8), dtype=np.uint8)
for i in range(8):
    r[:, :, i] = cv2.bitwise_and(img, x[:, :, i])
    mask = r[:, :, i] > 0
    r[mask] = 255
    cv2.imshow(str(i), r[:, :, i])
cv2.waitKey()
cv2.destroyAllWindows()

# two_picture('back.jpg', 'flowers.jpg')
# fun_f(200, 400, 500, 700, 'back.jpg', 'flowers.jpg')
# mask('flowers.jpg')
