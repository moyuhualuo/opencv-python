import cv2
import numpy as np
"""
>>>0.Get your area of interest
"""


def write(img, save_address):
    cv2.imwrite(save_address, img)


def light(address, add_light=10):
    img = cv2.imread(address)
    light_img = cv2.add(img, add_light)
    img_and_light = np.hstack((img, light_img))
    # compare them
    cv2.imshow('img_and_light', img_and_light)
    cv2.waitKey()
    cv2.destroyAllWindows()


def license_plate(address, a, b, c, d, w=None, write_file=None):
    """a, b, c, d means area of interest.
    write img, use w, 0 or None means don't write, 1 means write.
    If w equal 1, we must have correct write_file to save.
    """
    # 0 means black and white.
    img = cv2.imread(address, 0)
    # Get your area of interest, here are plate.
    plate = img[a:b, c:d]
    cv2.imshow('plate', plate)
    if w and write_file:
        write(plate, write_file)
    cv2.waitKey()
    cv2.destroyAllWindows()


'''Drop '#' to use these function, but first you should know function what to do'''
'''Don't write'''
# license_plate('track.png', 467, 500, 447, 542)

'''write'''
# license_plate('track.png', 467, 500, 447, 542, True, 'output_plate.png')

'''light img '''
# light('output_plate.png', -30)
