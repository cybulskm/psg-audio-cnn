import os
# FORCE CPU USAGE - CRITICAL FOR GPU-LESS SYSTEMS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
import multiprocessing as mp
from multiprocessing import Pool
import gc
import time
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_path': "/raid/userdata/cybulskm/ThesisProj/processed.pkl",
    'output_dir': "bin",
    'checkpoint_file': "bin/experiment_checkpoint.json",
    'results_file': "bin/cnn_feature_selection_results.csv",
    'n_runs': 5,
    'chunk_size': 5000,  # Smaller chunks for better memory management
    'n_processes': min(8, mp.cpu_count() - 4),  # Conservative process count
    'channels': ["Leg 2", "Leg 1", "EEG C3-A2", "EEG C4-A1", "EMG Chin",
                 "EEG A1-A2", "EOG LOC-A2", "EOG ROC-A2", "ECG I"]
}

def setup_cpu():
    """Configure for CPU usage only"""
    print("=" * 70)
    print("CONFIGURED FOR CPU-ONLY EXECUTION")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print("Available devices:", [d.device_type for d in tf.config.get_visible_devices()])
    print("CPU threads available:", mp.cpu_count())
    print("=" * 70)
    return None

def create_directory_structure():
    """Create necessary directories"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Created output directory: {CONFIG['output_dir']}")

class ProgressTracker:
    """Track progress and estimate completion time"""
    def __init__(self, total_steps, description=""):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.description = description

    def update(self, step=1, message=""):
        self.current_step += step
        elapsed = time.time() - self.start_time
        if self.current_step > 0:
            steps_per_second = self.current_step / elapsed
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0
            eta_str = str(timedelta(seconds=int(eta_seconds)))
        else:
            eta_str = "Calculating..."

        progress_pct = (self.current_step / self.total_steps) * 100
        print(f"\r[{self.description}] Progress: {self.current_step}/{self.total_steps} "
              f"({progress_pct:.1f}%) ETA: {eta_str} {message}",
              end="", flush=True)

    def complete(self):
        total_time = time.time() - self.start_time
        print(f"\n[{self.description}] Completed in {timedelta(seconds=int(total_time))}")

def process_segment_chunk(args):
    """Process a chunk of segments (for multiprocessing)"""
    chunk_idx, chunk, channels = args
    X_chunk = []
    y_chunk = []

    for seg in chunk:
        try:
            channel_data = []
            valid_segment = True

            for ch in channels:
                if ch in seg and seg[ch] is not None and len(seg[ch]) > 0:
                    data = np.array(seg[ch], dtype=np.float32)
                    if np.any(np.isnan(data)):
                        data = np.nan_to_num(data, nan=0.0)
                    channel_data.append(data)
                else:
                    valid_segment = False
                    break

            if valid_segment and len(channel_data) == len(channels):
                # Ensure all channels have same length
                min_len = min(len(ch) for ch in channel_data)
                if min_len > 10:  # Reasonable minimum length
                    channel_data = [ch[:min_len] for ch in channel_data]
                    X_chunk.append(np.array(channel_data).T)
                    y_chunk.append(seg.get('Label', 'Unknown'))

        except Exception as e:
            continue

    return chunk_idx, X_chunk, y_chunk


def process_segment_for_loading(args):
    """Process a single segment (moved outside for pickling)"""
    segment, channels = args
    try:
        channel_data = []
        valid_segment = True

        for ch in channels:
            if ch in segment and segment[ch] is not None and len(segment[ch]) > 0:
                data = np.array(segment[ch], dtype=np.float32)
                if np.any(np.isnan(data)):
                    data = np.nan_to_num(data, nan=0.0)
                channel_data.append(data)
            else:
                valid_segment = False
                break

        if valid_segment and len(channel_data) == len(channels):
            # Ensure all channels have same length
            min_len = min(len(ch) for ch in channel_data)
            if min_len > 10:  # Reasonable minimum length
                channel_data = [ch[:min_len] for ch in channel_data]
                return np.array(channel_data).T, segment.get('Label', 'Unknown')

    except Exception:
        pass
    
    return None, None

def load_data_optimized(data_path, channels):
    """Load data - handle both individual segments and list of segments"""
    print("Loading data...")
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the pickle file
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Check if it's a list of segments or individual segments
    if isinstance(data, list) and len(data) > 0:
        # It's a list of segments (what we actually have)
        segments = data
        print(f"Loaded list with {len(segments)} segments")
    else:
        # It's individual segments (old format)
        segments = [data]
        try:
            while True:
                segment = pickle.load(f)
                segments.append(segment)
        except EOFError:
            pass
        print(f"Loaded {len(segments)} individual segments")
    
    if len(segments) == 0:
        raise ValueError("No segments found in pickle file!")
    
    # Debug: Check first few segments
    print("Debugging first 3 segments:")
    for i, seg in enumerate(segments[:3]):
        if isinstance(seg, dict):
            print(f"Segment {i}: Keys = {list(seg.keys())}")
            print(f"Segment {i}: Label = {seg.get('Label', 'NO_LABEL')}")
            for ch in channels[:3]:  # Check first 3 channels
                if ch in seg:
                    data = seg[ch]
                    if data is not None and len(data) > 0:
                        print(f"  {ch}: length={len(data)}, type={type(data)}, sample={data[:5] if len(data) >= 5 else data}")
                    else:
                        print(f"  {ch}: EMPTY or None")
                else:
                    print(f"  {ch}: NOT FOUND")
        else:
            print(f"Segment {i}: Not a dictionary! Type = {type(seg)}")
    
    # Process segments
    X_all = []
    y_all = []
    valid_count = 0
    invalid_count = 0
    
    print(f"Processing {len(segments)} segments...")
    
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            invalid_count += 1
            continue
            
        X_data, y_data = process_segment_for_loading((seg, channels))
        if X_data is not None:
            X_all.append(X_data)
            y_all.append(y_data)
            valid_count += 1
        else:
            invalid_count += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(segments)} - Valid: {valid_count}, Invalid: {invalid_count}")
    
    print(f"Final counts - Valid: {valid_count}, Invalid: {invalid_count}")
    
    if not X_all:
        raise ValueError(f"No valid segments processed! All {invalid_count} segments were invalid.")
    
    # Convert to arrays
    print("Converting to numpy arrays...")
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)
    
    print(f"Final data shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")
    
    return X, y

def save_checkpoint(state):
    """Save current experiment state"""
    try:
        with open(CONFIG['checkpoint_file'], 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Checkpoint saved: {CONFIG['checkpoint_file']}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint():
    """Load experiment state from checkpoint"""
    if os.path.exists(CONFIG['checkpoint_file']):
        try:
            with open(CONFIG['checkpoint_file'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return None

class TimeTrackingCallback(Callback):
    """Custom callback to track training time"""
    def __init__(self, run_id, model_type):
        super().__init__()
        self.run_id = run_id
        self.model_type = model_type
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"  Starting training for {self.model_type}...")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Print every 5 epochs
            elapsed = time.time() - self.start_time
            print(f"  {self.model_type} - Epoch {epoch+1}/15 - Time: {timedelta(seconds=int(elapsed))}")

    def on_train_end(self, logs=None):
        training_time = time.time() - self.start_time
        print(f"  {self.model_type} training completed in {timedelta(seconds=int(training_time))}")

def get_feature_importance(X_train, y_train, channels):
    """Get feature importance using Random Forest"""
    print("Training Random Forest for feature importance...")

    # Convert to 2D for Random Forest (mean across time dimension)
    X_rf = np.mean(X_train, axis=1)  # Shape: (samples, channels)

    # Encode labels for RF
    y_rf = np.argmax(y_train, axis=1)

    print(f"RF input shape: {X_rf.shape}")
    print(f"RF labels: {np.unique(y_rf, return_counts=True)}")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=1)
    rf.fit(X_rf, y_rf)

    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = sorted(zip(channels, importances), key=lambda x: x[1], reverse=True)

    print("\nFeature importances (sorted):")
    for feature, importance in feature_importance:
        print(f"{feature}: {importance:.4f}")

    return feature_importance

def create_cnn_model(input_shape, num_classes):
    """Create 1D CNN model optimized for CPU"""
    model = Sequential([
        Conv1D(32, kernel_size=7, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_improved_cnn(input_shape, num_classes):
    """Enhanced CNN for sleep apnea detection"""
    model = Sequential([
        # First block - capture short-term patterns
        Conv1D(32, kernel_size=11, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.2),
        
        # Second block - medium-term patterns  
        Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        
        # Third block - long-term patterns
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Global pooling instead of flatten (handles variable lengths better)
        GlobalAveragePooling1D(),
        
        # Dense layers with proper regularization
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Use lower learning rate and better optimizer
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def get_training_callbacks():
    """Get callbacks for better training"""
    callbacks = [
        # Stop if no improvement for 15 epochs
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate if stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        )
    ]
    return callbacks


def load_data_streaming(data_path, channels, max_segments=None):
    """Memory-efficient streaming loader"""
    print("Loading data with streaming...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load segments count first
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        segments = data
        total_segments = len(segments)
    else:
        raise ValueError("Expected list format")
    
    if max_segments:
        segments = segments[:max_segments]
        total_segments = len(segments)
    
    print(f"Processing {total_segments} segments in streaming mode...")
    
    # Process in small batches to control memory
    batch_size = 100  # Very small batches
    X_batches = []
    y_batches = []
    
    for i in range(0, total_segments, batch_size):
        batch = segments[i:i+batch_size]
        X_batch = []
        y_batch = []
        
        for seg in batch:
            if not isinstance(seg, dict):
                continue
                
            # Process segment
            channel_data = []
            valid = True
            
            for ch in channels:
                if ch in seg and seg[ch] is not None:
                    data = np.array(seg[ch], dtype=np.float16)  # Use float16 to halve memory
                    if len(data) > 0:
                        channel_data.append(data)
                    else:
                        valid = False
                        break
                else:
                    valid = False
                    break
            
            if valid and len(channel_data) == len(channels):
                min_len = min(len(ch) for ch in channel_data)
                if min_len > 1000:  # Reasonable minimum
                    channel_data = [ch[:min_len] for ch in channel_data]
                    X_batch.append(np.array(channel_data, dtype=np.float16).T)
                    y_batch.append(seg.get('Label', 'Unknown'))
        
        if X_batch:
            X_batches.append(np.array(X_batch, dtype=np.float16))
            y_batches.extend(y_batch)
        
        # Clear batch from memory
        del batch, X_batch
        gc.collect()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i+batch_size}/{total_segments} segments")
    
    # Clear original segments
    del segments, data
    gc.collect()
    
    # Combine batches
    print("Combining batches...")
    X = np.concatenate(X_batches, axis=0)
    y = np.array(y_batches)
    
    print(f"Final shape: {X.shape}, Memory: {X.nbytes / 1e9:.1f} GB")
    return X, y


def validate_data_quality(X, y):
    """Check for data quality issues that hurt CNN performance"""
    print("\n🔍 DATA QUALITY CHECK:")
    print("-" * 40)
    
    # Check for data imbalance
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Check for extreme imbalance
    imbalance_ratio = max(counts) / min(counts)
    if imbalance_ratio > 10:
        print(f"⚠️  SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}:1")
        print("   Consider using class weights or SMOTE")
    
    # Check data ranges
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Data std: {X.std():.3f}")
    
    # Check for NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Data issues: {nan_count} NaNs, {inf_count} Infs")
    
    return imbalance_ratio

def preprocess_for_cnn(X, y):
    """Preprocess data for better CNN performance"""
    print("📊 PREPROCESSING FOR CNN:")
    
    # 1. Standardize the data (critical for CNNs)
    print("Standardizing data...")
    X_std = np.copy(X)
    for i in range(X.shape[-1]):  # Standardize each channel
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
    
    # 2. Handle class imbalance
    unique_labels, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
    imbalance_ratio = max(counts) / min(counts)
    
    if imbalance_ratio > 3:
        print(f"Computing class weights for imbalance ratio: {imbalance_ratio:.1f}")
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=np.argmax(y, axis=1)
        )
        class_weight_dict = dict(zip(unique_labels, class_weights))
        print(f"Class weights: {class_weight_dict}")
        return X_std, y, class_weight_dict
    
    return X_std, y, None

def process_large_dataset_efficiently():
    """Improved main function with better CNN training"""
    print("=" * 70)
    print("ENHANCED CNN MODE - OPTIMIZED FOR LEARNING")
    print("=" * 70)
    
    # Load data
    print("Loading subset for training...")
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], max_segments=1000)
    
    # Validate data quality
    labels, y_encoded = np.unique(y, return_inverse=True)
    print(f"Labels found: {labels}")
    imbalance_ratio = validate_data_quality(X, y_encoded)
    
    if len(labels) < 2:
        raise ValueError("Need at least 2 classes for classification!")
    
    # Convert labels and split
    y_cat = to_categorical(y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Preprocess data for CNN
    X_train_processed, y_train_processed, class_weights = preprocess_for_cnn(X_train, y_train)
    X_test_processed = X_test.copy()
    
    # Standardize test set using training statistics
    for i in range(X_test.shape[-1]):
        train_mean = np.mean(X_train[:, :, i])
        train_std = np.std(X_train[:, :, i])
        if train_std > 0:
            X_test_processed[:, :, i] = (X_test[:, :, i] - train_mean) / train_std
    
    print(f"Training shape: {X_train_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")
    
    # Feature importance (keep your existing code)
    print("Computing feature importance...")
    X_rf = np.mean(X_train_processed, axis=1).astype(np.float32)
    y_rf = np.argmax(y_train_processed, axis=1)
    
    rf = RandomForestClassifier(
        n_estimators=50,  # Increased for better stability
        max_depth=15,
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_rf, y_rf)
    
    importances = rf.feature_importances_
    feature_importance = sorted(zip(CONFIG['channels'], importances), key=lambda x: x[1], reverse=True)
    
    print("Feature importances:")
    for feature, importance in feature_importance:
        print(f"  {feature}: {importance:.4f}")
    
    # Select top features
    top_features = min(6, len(CONFIG['channels']))  # Use top 6 features
    feature_indices = [CONFIG['channels'].index(feat) for feat, _ in feature_importance[:top_features]]
    
    X_train_subset = X_train_processed[:, :, feature_indices] 
    X_test_subset = X_test_processed[:, :, feature_indices]
    
    print(f"\nUsing top {top_features} features: {[feat for feat, _ in feature_importance[:top_features]]}")
    
    # Create improved CNN
    model = create_improved_cnn(X_train_subset.shape[1:], y_train_processed.shape[1])
    
    print(f"\nModel summary:")
    model.summary()
    
    # Train with improved setup
    callbacks = get_training_callbacks()
    
    print("\n🚀 Starting enhanced CNN training...")
    history = model.fit(
        X_train_subset, y_train_processed,
        epochs=100,  # More epochs with early stopping
        batch_size=32,  # Larger batch size for stability
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weights,  # Handle imbalance
        verbose=1
    )
    
    # Evaluate
    print("\n📊 FINAL EVALUATION:")
    loss, acc = model.evaluate(X_test_subset, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    
    # Show training history
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = max(history.history['val_accuracy'])
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
    
    return acc, feature_importance

# Replace main() with:
def main():
    """Main function with enhanced CNN training"""
    setup_cpu()
    
    try:
        acc, feature_importance = process_large_dataset_efficiently()
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Test Accuracy: {acc:.4f}")
        print("Top 3 features:", [f[0] for f in feature_importance[:3]])
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_directory_structure()
    main()
