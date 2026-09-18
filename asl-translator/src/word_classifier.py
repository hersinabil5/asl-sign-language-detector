import json
import numpy as np
from tensorflow import keras


class WordClassifier:
    """Wraps the trained LSTM word classifier. Knows nothing about
    webcams or landmark extraction — just takes a sequence of
    landmark frames and returns the predicted word + confidence."""

    def __init__(self, model_path='models/word_classifier.h5', classes_path='models/word_classes.json'):
        self._model = keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self._classes = json.load(f)

    def predict(self, sequence):
        """sequence: list of frames, each a list of 63 floats
        (matching training: flattened x,y,z for 21 landmarks).
        Returns (word, confidence)."""
        input_array = np.array([sequence])  # shape: (1, max_frames, 63)
        probabilities = self._model.predict(input_array, verbose=0)[0]
        best_idx = int(np.argmax(probabilities))
        return self._classes[best_idx], float(probabilities[best_idx])
