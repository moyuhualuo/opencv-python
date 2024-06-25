import time
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # Open webcam with index 0 (usually the built-in webcam)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame from camera. Exiting...")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image
            for id, lm in enumerate(hand_landmarks.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                #if id == 4:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Display the image with landmarks


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # Exit if 'q' is pressed

    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break



# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


