import sys
import cv2
import mediapipe as mp
import json
# 初始化 MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# 初始化一个空的 Hand_dict
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


def detect_hand_gesture(image_path):
    # 读取静态图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from path: {image_path}")
        return

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 处理图像并获取结果
    results = hands.process(imgRGB)

    # 检查是否检测到手部
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 提取手指状态字符串
            finger_states = []
            for id, lm in enumerate(hand_landmarks.landmark):
                if id in [4, 8, 12, 16, 20]:  # 指尖的索引（食指、中指、无名指、小指、拇指）
                    # 获取手指关键点在图像中的位置
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    # 判断手指是否伸直（假设手指处于某个阈值附近时认为伸直）
                    if lm.y < 0.8:  # 根据需要调整阈值
                        finger_states.append('1')
                    else:
                        finger_states.append('0')

            # 将手指状态列表转换为状态字符串
            gesture = ''.join(finger_states)

            # 判断手势是否在 Hand_dict 中
            if gesture in Hand_dict:
                gesture_name = Hand_dict[gesture]
                print(f'Detected Gesture: {gesture_name}')
                # 在图像上绘制手势名称
                cv2.putText(img, f'Gesture: {gesture_name}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                print('Unknown Gesture')

    # 显示处理后的图像
    cv2.imshow('Hand Gesture Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python HandImage.py <image_path>")
    else:
        image_path = sys.argv[1]
        detect_hand_gesture(image_path)
