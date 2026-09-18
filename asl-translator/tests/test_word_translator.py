import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from collections import deque
from src.hand_tracker import HandTracker
from src.draw_utils import draw_landmarks
from src.word_classifier import WordClassifier

MAX_FRAMES = 40
PREDICT_EVERY = 10  # run a new prediction every 10 frames, not every single frame

tracker = HandTracker(model_path='models/hand_landmarker.task', num_hands=2)
classifier = WordClassifier(model_path='models/word_classifier.h5', classes_path='models/word_classes.json')

buffer = deque(maxlen=MAX_FRAMES)
frame_count = 0
last_prediction = "..."
last_confidence = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hands = tracker.detect(frame)
    if hands:
        flat = [coord for point in hands[0] for coord in point]
        draw_landmarks(frame, hands)
    else:
        flat = [0.0] * 63

    buffer.append(flat)
    frame_count += 1

    if len(buffer) == MAX_FRAMES and frame_count % PREDICT_EVERY == 0:
        word, confidence = classifier.predict(list(buffer))
        last_prediction = word
        last_confidence = confidence
        print(f"Prediction: {word} ({confidence:.1%})")

    cv2.putText(frame, f'{last_prediction} ({last_confidence:.0%})', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Word Translator Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
