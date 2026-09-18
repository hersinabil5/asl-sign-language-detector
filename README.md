# ASL Word Recognition

A real-time American Sign Language word recognition system, built from the ground up: custom hand-tracking pipeline, landmark normalization, LSTM-based sequence modeling, and live webcam inference.

## Overview

Recognizes a 20-word ASL vocabulary from live video using a sequence model trained on the WLASL (Word-Level ASL) dataset — the same benchmark dataset used in ASL recognition research.

**Vocabulary:**
`before, cool, thin, drink, go, who, help, cousin, computer, tall, bed, thanksgiving, candy, short, accident, bowling, shirt, man, yes, basketball`

## Results

| Metric | Result |
|---|---|
| Validation accuracy | 55-60% across 20 classes |
| Performance vs. random baseline | 10-12x |
| Model | 2-layer LSTM, 153K parameters |
| Training data | 498 hand-tracked video sequences |

## Pipeline

Webcam frame
│
▼
Custom HandTracker (MediaPipe Tasks Vision)
│
▼
Landmark normalization — wrist-centered, scale-invariant
│
▼
40-frame rolling sequence buffer
│
▼
LSTM sequence classifier ──► predicted word + confidence


Every stage — hand tracking, normalization, sequence buffering, and classification — is built as an independent, testable component (`src/`), not a single monolithic script.

## Tech Stack

**Computer vision:** MediaPipe Tasks Vision, OpenCV
**Machine learning:** TensorFlow / Keras (LSTM), scikit-learn
**Data:** WLASL dataset, NumPy for sequence processing

## File Structure

asl-translator/
├── app.py
├── requirements.txt
├── collect_data.py
├── explore_dataset.py
├── preprocess_videos.py
├── train_word_classifier.py
├── models/
│ ├── hand_landmarker.task
│ ├── word_classifier.h5
│ └── word_classes.json
├── src/
│ ├── hand_tracker.py # Hand-landmark detection
│ ├── draw_utils.py # Skeleton visualization
│ ├── landmark_utils.py # Normalization pipeline
│ ├── dataset_collector.py # Training data collection tooling
│ └── word_classifier.py # LSTM inference wrapper
├── data/
├── raw_dataset/
└── tests/
├── test_hand_tracker.py
└── test_word_translator.py


## Setup

```bash
git clone https://github.com/hersinabil5/asl-sign-language-detector.git
cd asl-sign-language-detector/asl-translator
python3 -m venv venv
source venv/bin/activate
pip install mediapipe opencv-python tensorflow scikit-learn numpy
mkdir -p models
curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

## Usage

```bash
python3 tests/test_hand_tracker.py       # Live hand tracking demo
python3 tests/test_word_translator.py    # Live word recognition
```

To retrain on the full pipeline:

```bash
python3 explore_dataset.py
python3 preprocess_videos.py
python3 train_word_classifier.py
```

## Roadmap

- Web interface via Flask + WebSockets for browser-based recognition
- Expanded vocabulary with additional training data
- Two-handed sign support
- Face and pose tracking for grammar-relevant signs
