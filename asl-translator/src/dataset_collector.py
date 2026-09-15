import csv
import os


class DatasetCollector:
    """Handles writing labeled landmark samples to a CSV file.
    Knows nothing about cameras, MediaPipe, or classification —
    just takes landmarks + a label and saves them."""

    def __init__(self, csv_path='data/asl_landmarks.csv'):
        self._csv_path = csv_path
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        os.makedirs(os.path.dirname(self._csv_path), exist_ok=True)
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 21 landmarks × (x, y, z) = 63 columns, plus the label
                header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')]
                header.append('label')
                writer.writerow(header)

    def save_sample(self, landmarks, label):
        """landmarks: list of 21 (x, y, z) tuples. label: e.g. 'A'."""
        flat = [coord for point in landmarks for coord in point]
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(flat + [label])

    def sample_count(self):
        """Returns how many samples have been collected so far."""
        if not os.path.exists(self._csv_path):
            return 0
        with open(self._csv_path, 'r') as f:
            return sum(1 for _ in f) - 1  # minus header row
