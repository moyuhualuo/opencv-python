import cv2


def show_picture(name, src_address):
    cv2.namedWindow(name)
    src = cv2.imread(src_address, -1)
    '''
    >>> -1 means the same as before, 0 means Grayscale, 1 means BGR.
    >>> what about None same, -1, 0 or 1? 
        -1
    '''
    cv2.imshow(name, src)
    key = cv2.waitKey()
    '''
    >>> input key to close window
    '''
    if key != -1:
        print('close the window')
        cv2.destroyWindow(name)
    '''
    >>> free memory
    '''


def save_picture(src_address, save_address):
    src = cv2.imread(src_address)
    r = cv2.imwrite(save_address, src)


'''
>>> What can we do use these easy functions?  
'''

show_picture('No_name', r"D:\opencv\lab01\brush-strokes-6139429_1280.png")
save_picture(R"D:\opencv\lab01\brush-strokes-6139429_1280.png", 'save_address')