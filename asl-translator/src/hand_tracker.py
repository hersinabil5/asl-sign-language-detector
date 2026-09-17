import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    """Wraps MediaPipe's hand landmark detection. Knows nothing about
    ASL, classification, or Flask — just turns a frame into landmarks."""

    def __init__(self, model_path='models/hand_landmarker.task', num_hands=2, min_confidence=0.7):
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        """Takes a BGR OpenCV frame, returns a list of hands, each a
        list of 21 (x, y, z) normalized landmark tuples. Empty list if
        no hand detected."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        hands = []
        for hand_landmarks in result.hand_landmarks:
            hands.append([(lm.x, lm.y, lm.z) for lm in hand_landmarks])
        return hands
