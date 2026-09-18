import cv2
import json
import os
import numpy as np
from src.hand_tracker import HandTracker
from src.landmark_utils import normalize_landmarks

# Top 20 words by available video count (from explore_dataset.py)
WORD_LIST = [
    'before', 'cool', 'thin', 'drink', 'go', 'who', 'help', 'cousin', 'computer', 'tall',
    'bed', 'thanksgiving', 'candy', 'short', 'accident', 'bowling', 'shirt', 'man', 'yes', 'basketball'
]

MAX_FRAMES = 40

with open('raw_dataset/WLASL_v0.3.json', 'r') as f:
    wlasl_data = json.load(f)

word_to_videos = {}
for entry in wlasl_data:
    if entry['gloss'] in WORD_LIST:
        word_to_videos[entry['gloss']] = [inst['video_id'] for inst in entry['instances']]

tracker = HandTracker(model_path='models/hand_landmarker.task', num_hands=2)

sequences = []
labels = []

for word in WORD_LIST:
    video_ids = word_to_videos.get(word, [])
    processed_count = 0

    for video_id in video_ids:
        video_path = f'raw_dataset/videos/{video_id}.mp4'
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frame_landmarks = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            hands = tracker.detect(frame)
            if hands:
                flat = [coord for point in hands[0] for coord in point]
                flat = normalize_landmarks(flat)
            else:
                flat = [0.0] * 63
            frame_landmarks.append(flat)

        cap.release()

        if len(frame_landmarks) == 0:
            continue

        if len(frame_landmarks) > MAX_FRAMES:
            frame_landmarks = frame_landmarks[:MAX_FRAMES]
        else:
            while len(frame_landmarks) < MAX_FRAMES:
                frame_landmarks.append([0.0] * 63)

        sequences.append(frame_landmarks)
        labels.append(word)
        processed_count += 1

    print(f"'{word}': processed {processed_count} videos")

sequences = np.array(sequences)
labels = np.array(labels)

os.makedirs('data', exist_ok=True)
np.savez('data/word_sequences.npz', sequences=sequences, labels=labels)

print(f"\nTotal samples: {len(sequences)}")
print(f"Sequence shape: {sequences.shape}")
