import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

data = np.load('data/word_sequences.npz', allow_pickle=True)
sequences = data['sequences']
labels = data['labels']

le = LabelEncoder()
y_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = keras.Sequential([
    keras.layers.Masking(mask_value=0.0, input_shape=(40, 63)),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.Dropout(0.3),
    keras.layers.LSTM(64),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

model.save('models/word_classifier.h5')

import json
with open('models/word_classes.json', 'w') as f:
    json.dump(le.classes_.tolist(), f)

print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.2%}")