import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import json
from datetime import datetime
import random

# Channel rankings (from previous analysis)
CHANNEL_RANKING = [
    ('ECG I', 0.128733),
    ('EEG C4-A1', 0.105451),
    ('EOG ROC-A2', 0.105369),
    ('EEG A1-A2', 0.105110),
    ('EOG LOC-A2', 0.102342),
    ('EEG C3-A2', 0.100272),
    ('Leg 2', 0.096922),
    ('Leg 1', 0.096168),
    ('EMG Chin', 0.092785)
]

def create_simple_model(input_shape):
    """Create a simplified 1D CNN model"""
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')  # 5 sleep stages
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(X, y, run_name):
    """Train the model with early stopping"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = create_simple_model((X.shape[1], X.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc, model, history

def run_feature_tests(X, y, n_runs=5):
    """Run tests for different channel selections"""
    results = {}
    all_channels = [ch for ch, _ in CHANNEL_RANKING]
    
    # Define channel groups
    groups = {
        'top_25': all_channels[:3],
        'top_50': all_channels[:5],
        'random': lambda: random.sample(all_channels, 5)
    }
    
    for group_name, channels in groups.items():
        print(f"\nTesting {group_name}...")
        group_results = []
        
        for run in range(n_runs):
            # Get channels (handle random selection)
            selected_channels = channels() if callable(channels) else channels
            
            # Select channel data
            channel_indices = [all_channels.index(ch) for ch in selected_channels]
            X_subset = X[:, :, channel_indices]
            
            # Train and evaluate
            start_time = time.time()
            accuracy, _, _ = train_model(X_subset, y, f"{group_name}_run_{run}")
            run_time = time.time() - start_time
            
            result = {
                'run': run + 1,
                'accuracy': float(accuracy),
                'time': run_time,
                'channels': selected_channels
            }
            group_results.append(result)
            print(f"Run {run+1}: Accuracy = {accuracy:.4f}, Time = {run_time:.1f}s")
            
        results[group_name] = group_results
    
    return results

def save_results(results):
    """Save test results to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'cnn_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    """Main test function"""
    print("Loading data...")
    # Replace with your data loading code
    # X should be shape (n_samples, timesteps, channels)
    # y should be one-hot encoded labels
    
    results = run_feature_tests(X, y)
    save_results(results)
    
    # Print summary
    print("\nTest Summary:")
    for group, runs in results.items():
        accuracies = [r['accuracy'] for r in runs]
        times = [r['time'] for r in runs]
        print(f"\n{group}:")
        print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean time: {np.mean(times):.1f}s")

if __name__ == "__main__":
    main()