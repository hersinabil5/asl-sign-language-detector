import cv2
from src.hand_tracker import HandTracker
from src.dataset_collector import DatasetCollector

tracker = HandTracker(model_path='models/hand_landmarker.task')
collector = DatasetCollector(csv_path='data/asl_landmarks.csv')

print("Press a letter key (A-Z) to save the current hand shape as that label.")
print("Press 'q' to quit.")
print(f"Samples so far: {collector.sample_count()}")

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    hands = tracker.detect(frame)

    if hands:
        h, w, _ = frame.shape
        for x, y, z in hands[0]:
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    cv2.putText(frame, f'Samples: {collector.sample_count()}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Data Collection', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif hands and chr(key).isalpha() and key != 255:
        label = chr(key).upper()
        collector.save_sample(hands[0], label)
        print(f"Saved sample for '{label}' (total: {collector.sample_count()})")

cap.release()
cv2.destroyAllWindows()
