import time
import cv2
import mediapipe as mp
import json

Hand_dict = {}
# 加载保存的 Hand_dict 文件
def load_hand_dict(filename):
    global Hand_dict
    try:
        with open(filename, 'r') as f:
            Hand_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

load_hand_dict('hand_dict.json')

"""
['00001', '11011', '00110', '11101', '10101', '10001', '11000', '10100', '11110', '11001', '10011', '10111', '00011']
可以自定义设置手势！！！！
"""
cap = cv2.VideoCapture(0)  # 打开摄像头

# 初始化 MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

def detect_hand_gesture(hand_landmarks):
    lmList = []
    for id, lm in enumerate(hand_landmarks.landmark):
        lmList.append([id, lm.x, lm.y])

    if len(lmList) != 0:
        fingers = []

        # 拇指
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)  # 完全伸直
        else:
            fingers.append(0)  # 未完全伸直

        # 其余四个手指
        for id in range(8, 21, 4):
            if lmList[id][2] < lmList[id - 2][2]:
                fingers.append(1)  # 完全伸直
            else:
                fingers.append(0)  # 未完全伸直

        # 返回手指状态
        return fingers
    return [0, 0, 0, 0, 0]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Exiting...")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
            fingers = detect_hand_gesture(hand_landmarks)
            gesture = ''.join(map(str, fingers))  # 将手指状态转化为字符串
            cv2.putText(img, f'GestureCode: {gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            if gesture in Hand_dict:
                cv2.putText(img, f'Gesture: {Hand_dict[gesture]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            else:
                cv2.putText(img, f'Gesture: None', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255),
                            2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()