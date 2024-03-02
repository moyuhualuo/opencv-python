import cv2
import datetime

def cap():
    cap0 = cv2.VideoCapture(0)  # init cap

    while cap0.isOpened():
        ret, frame = cap0.read()
        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c == 27:  # 'Esc'
            break
    cap0.release()
    cv2.destroyAllWindows()


def video():
    cap = cv2.VideoCapture('clock_-_28723 (720p).mp4')
    while cap.isOpened():
        """if or while, they are different."""
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        c = cv2.waitKey(25)
        if c == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def save_cap():
    cap1 = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(f'output.mp4', fourcc, 20.0, (640, 480))
    while cap1.isOpened():
        ret, frame = cap1.read()
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:
                break
        else:
            break
    cap1.release()
    out.release()
    cv2.destroyAllWindows()

def canny
    mp4 = cv2.VideoCapture('clock_-_28723 (720p).mp4')
    while mp4.isOpened():
        ret, frame = mp4.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(frame, 50, 50)
            cv2.imshow('frame', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
    mp4.release()
    cv2.destroyAllWindows()
