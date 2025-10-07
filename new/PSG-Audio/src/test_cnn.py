import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time
import json
from datetime import datetime
import random
import tensorflow as tf

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import project modules
from src.data_loader import load_data_streaming, validate_data_quality
from config.config import CONFIG

# Channel rankings
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

def setup_cpu_environment():
    """Setup CPU-only environment"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.config.set_visible_devices([], 'GPU')
    print(f"TensorFlow: {tf.__version__} (CPU-only)")

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = create_simple_model((X.shape[1], X.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.2,
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return test_acc, model, history

def run_feature_tests(X, y, n_runs=5):
    """Run tests for different channel selections"""
    results = {}
    all_channels = [ch for ch, _ in CHANNEL_RANKING]
    
    groups = {
        'top_25': all_channels[:3],
        'top_50': all_channels[:5],
        'random': lambda: random.sample(all_channels, 5)
    }
    
    for group_name, channels in groups.items():
        print(f"\nTesting {group_name}...")
        group_results = []
        
        for run in range(n_runs):
            selected_channels = channels() if callable(channels) else channels
            channel_indices = [all_channels.index(ch) for ch in selected_channels]
            X_subset = X[:, :, channel_indices]
            
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
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    filename = os.path.join(CONFIG['output_dir'], f'cnn_results_{timestamp}.json')
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

def main():
    """Main test function"""
    setup_cpu_environment()
    print("\n=== Simple CNN Feature Selection Test ===")
    print("Loading data...")
    
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'])
    labels, y_encoded = np.unique(y, return_inverse=True)
    y_cat = to_categorical(y_encoded)
    
    print(f"Data loaded: X shape={X.shape}, unique labels={labels}")
    validate_data_quality(X, y_encoded)
    
    results = run_feature_tests(X, y_cat)
    save_results(results)
    
    print("\nTest Summary:")
    for group, runs in results.items():
        accuracies = [r['accuracy'] for r in runs]
        times = [r['time'] for r in runs]
        print(f"\n{group}:")
        print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Mean time: {np.mean(times):.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")