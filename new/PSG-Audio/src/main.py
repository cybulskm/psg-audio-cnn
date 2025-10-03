import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.data_loader import load_data_streaming
from src.random_forest import get_feature_importance, select_top_features
from src.cnn import create_improved_cnn, train_and_evaluate_cnn
from config.config import CONFIG

def main():
    """Main function to orchestrate the workflow"""
    print("=" * 70)
    print("PSG-Audio: Random Forest and CNN Workflow")
    print("=" * 70)

    # Load data
    print("Loading data...")
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], max_segments=1000)

    # Validate data quality
    labels, y_encoded = np.unique(y, return_inverse=True)
    print(f"Labels found: {labels}")

    if len(labels) < 2:
        raise ValueError("Need at least 2 classes for classification!")

    # Convert labels to categorical
    y_cat = to_categorical(y_encoded)

    # Get feature importance from Random Forest
    feature_importance = get_feature_importance(X, y_cat, CONFIG['channels'])

    # Select top 25% and top 50% features
    top_25_percent_features = select_top_features(feature_importance, percentage=0.25)
    top_50_percent_features = select_top_features(feature_importance, percentage=0.50)

    # Prepare datasets for CNN
    X_train_25, y_train_25 = prepare_data_for_cnn(X, y_cat, top_25_percent_features)
    X_train_50, y_train_50 = prepare_data_for_cnn(X, y_cat, top_50_percent_features)

    # Train and evaluate CNN on top 25% features
    print("\nTraining CNN on top 25% features...")
    acc_25 = train_and_evaluate_cnn(X_train_25, y_train_25)

    # Train and evaluate CNN on top 50% features
    print("\nTraining CNN on top 50% features...")
    acc_50 = train_and_evaluate_cnn(X_train_50, y_train_50)

    print(f"\nFinal Results:")
    print(f"Accuracy with top 25% features: {acc_25:.4f}")
    print(f"Accuracy with top 50% features: {acc_50:.4f}")

def prepare_data_for_cnn(X, y, selected_features):
    """Prepare data for CNN based on selected features"""
    feature_indices = [CONFIG['channels'].index(feat) for feat in selected_features]
    X_subset = X[:, :, feature_indices]
    return X_subset, y

if __name__ == "__main__":
    main()