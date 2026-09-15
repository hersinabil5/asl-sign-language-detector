import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from src.hand_tracker import HandTracker

tracker = HandTracker(model_path='models/hand_landmarker.task')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hands = tracker.detect(frame)

    if hands:
        h, w, _ = frame.shape
        for hand in hands:
            for x, y, z in hand:
                px, py = int(x * w), int(y * h)
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
        print(f"Detected {len(hands)} hand(s)")

    cv2.imshow('HandTracker Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
