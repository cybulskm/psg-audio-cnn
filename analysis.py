import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load preprocessed data
data_path = os.path.join("bin", "processed.pkl")
with open(data_path, "rb") as f:
    segments = pickle.load(f)

# Prepare data
channels = ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", "EMG Chin", "Leg 1", "Leg 2", "ECG I"]
X = []
y = []
for seg in segments:
    # Stack channel data into shape (window_size, num_channels)
    channel_data = [seg[ch] if ch in seg else [np.nan]*len(seg[channels[0]]) for ch in channels]
    X.append(np.array(channel_data).T)  # shape: (window_size, num_channels)
    y.append(seg['Label'])

X = np.array(X)
# Encode labels
labels, y_encoded = np.unique(y, return_inverse=True)
y_cat = to_categorical(y_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build 1D CNN model
model = Sequential([
    Conv1D(32, kernel_size=7, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# Predict on test set
y_pred = model.predict(X_test)
y_pred_labels = labels[np.argmax(y_pred, axis=1)]
y_true_labels = labels[np.argmax(y_test, axis=1)]

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_true_labels, y_pred_labels))

print("Confusion Matrix:")
print(confusion_matrix(y_true_labels, y_pred_labels))

# Save model
model.save("bin/psg_cnn_model.h5")
print("Model saved!")