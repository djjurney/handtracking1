import cv2
import mediapipe as mp
import time
import tflite
import xnnpack

xnnpack_delegate = xnnpack.Delegate()

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    cv2.imshow("Image", img)
    cv2.waitKey(0)