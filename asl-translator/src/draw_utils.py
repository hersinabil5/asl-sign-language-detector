import cv2

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm base
]


def draw_landmarks(frame, hands, color=(0, 0, 255)):
    """hands: list of hands, each a list of (x, y, z) normalized tuples.
    Draws connecting lines and joint dots directly onto frame."""
    h, w, _ = frame.shape
    for hand in hands:
        points = [(int(x * w), int(y * h)) for x, y, z in hand]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], color, 2)
        for x, y in points:
            cv2.circle(frame, (x, y), 3, color, -1)
