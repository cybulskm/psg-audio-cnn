import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load preprocessed data
data_path = os.path.join("bin", "processed.pkl")
with open(data_path, "rb") as f:
    segments = pickle.load(f)

# Prepare data
channels = ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", "EMG Chin", "Leg 1", "Leg 2", "ECG I"]
X = []
y = []
for seg in segments:
    # For Random Forest, use mean of each channel in the segment as feature
    features = [np.nanmean(seg[ch]) if ch in seg else np.nan for ch in channels]
    X.append(features)
    y.append(seg['Label'])

X = np.array(X)
# Remove rows with NaN values
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = np.array(y)[mask]

# Encode labels
labels, y_encoded = np.unique(y, return_inverse=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

# Feature importance
importances = rf.feature_importances_
feature_importance = sorted(zip(channels, importances), key=lambda x: x[1], reverse=True)

print("\nFeature importances (sorted):")
for feature, importance in feature_importance:
    print(f"{feature}: {importance}")