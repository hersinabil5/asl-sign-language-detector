import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.hand_tracker import HandTracker
from src.draw_utils import draw_landmarks

tracker = HandTracker(model_path='models/hand_landmarker.task')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
last_hands = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    frame_count += 1
    if frame_count % 2 == 0:
        last_hands = tracker.detect(frame)

    if last_hands:
        draw_landmarks(frame, last_hands)

    cv2.imshow('HandTracker Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
