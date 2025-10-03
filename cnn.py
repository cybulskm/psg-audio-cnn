import os
# FORCE CPU USAGE - CRITICAL FOR GPU-LESS SYSTEMS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')
import gc
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'data_path': os.path.join("/raid/userdata/cybulskm/ThesisProj/", "100_patients_processed.pkl"),  # Use 100-patient dataset
    'output_file': "cnn_results_no_mixed.txt",
    'channels': ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", 
                 "EMG Chin", "Leg 1", "Leg 2", "ECG I"],
    'max_segments': None,  # Use all available data
    'test_size': 0.25,
    'random_state': 42,
    'min_class_samples': 100,  # Minimum samples per class
    'exclude_classes': ['MixedApnea']  # Classes to exclude
}

def log_message(message, file_path=None):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if file_path:
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

def setup_cpu():
    """Configure for CPU usage only"""
    log_message("=" * 70)
    log_message("CONFIGURED FOR CPU-ONLY EXECUTION")
    log_message("=" * 70)
    log_message(f"TensorFlow version: {tf.__version__}")
    log_message(f"Available devices: {[d.device_type for d in tf.config.get_visible_devices()]}")
    log_message("=" * 70)

def load_data_efficiently(data_path, channels, max_segments=None, exclude_classes=None):
    """Load data with memory management and class filtering"""
    log_message(f"Loading data from: {data_path}")
    log_message(f"Excluding classes: {exclude_classes}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load pickle file
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected list of segments in pickle file")
    
    log_message(f"Loaded {len(data)} segments from pickle file")
    
    # Process segments and filter out excluded classes
    X_list = []
    y_list = []
    valid_count = 0
    invalid_count = 0
    excluded_count = 0
    
    log_message("Processing segments and filtering classes...")
    
    for i, seg in enumerate(data):
        if not isinstance(seg, dict):
            invalid_count += 1
            continue
        
        # Check if this segment should be excluded
        label = seg.get('Label', 'Unknown')
        if exclude_classes and label in exclude_classes:
            excluded_count += 1
            continue
            
        # Check if all channels are present
        channel_data = []
        valid_segment = True
        
        for ch in channels:
            if ch in seg and seg[ch] is not None and len(seg[ch]) > 0:
                ch_data = np.array(seg[ch], dtype=np.float32)
                # Handle NaN values
                if np.any(np.isnan(ch_data)):
                    ch_data = np.nan_to_num(ch_data, nan=0.0)
                channel_data.append(ch_data)
            else:
                valid_segment = False
                break
        
        if valid_segment and len(channel_data) == len(channels):
            # Ensure all channels have same length
            min_len = min(len(ch) for ch in channel_data)
            if min_len > 100:  # Minimum viable segment length
                # Truncate to same length and transpose (samples x channels)
                channel_data = [ch[:min_len] for ch in channel_data]
                X_list.append(np.array(channel_data).T)  # Shape: (time_points, channels)
                y_list.append(label)
                valid_count += 1
            else:
                invalid_count += 1
        else:
            invalid_count += 1
        
        # Progress update
        if (i + 1) % 1000 == 0:
            log_message(f"  Processed {i+1}/{len(data)} - Valid: {valid_count}, Invalid: {invalid_count}, Excluded: {excluded_count}")
    
    log_message(f"Final processing results - Valid: {valid_count}, Invalid: {invalid_count}, Excluded: {excluded_count}")
    
    if not X_list:
        raise ValueError("No valid segments found!")
    
    # Limit segments for memory management if specified
    if max_segments and len(X_list) > max_segments:
        log_message(f"Limiting to {max_segments} segments for memory management")
        indices = np.random.choice(len(X_list), max_segments, replace=False)
        X_list = [X_list[i] for i in indices]
        y_list = [y_list[i] for i in indices]
    
    # Convert to numpy arrays
    log_message("Converting to numpy arrays...")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    log_message(f"Final data shape: {X.shape}")
    log_message(f"Memory usage: {X.nbytes / 1e6:.1f} MB")
    
    # Print class distribution after filtering
    unique_labels, counts = np.unique(y, return_counts=True)
    log_message("Class distribution after filtering:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(y)) * 100
        log_message(f"  {label}: {count} ({percentage:.1f}%)")
    
    return X, y

def preprocess_data(X, y):
    """Preprocess data for CNN training"""
    log_message("Preprocessing data for CNN...")
    
    # Encode labels
    unique_labels, y_encoded = np.unique(y, return_inverse=True)
    log_message(f"Label encoding: {dict(zip(unique_labels, range(len(unique_labels))))}")
    
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes for classification!")
    
    # Convert to categorical
    y_categorical = to_categorical(y_encoded)
    
    # Standardize features (critical for CNN performance)
    log_message("Standardizing features...")
    X_std = np.copy(X)
    
    for i in range(X.shape[-1]):  # For each channel
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
        log_message(f"  Channel {i} ({CONFIG['channels'][i]}): mean={mean:.3f}, std={std:.3f}")
    
    # Check for class imbalance
    unique_encoded, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip(unique_labels, counts))
    log_message(f"Class distribution: {class_distribution}")
    
    imbalance_ratio = max(counts) / min(counts)
    log_message(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Compute class weights if imbalanced
    class_weights = None
    if imbalance_ratio > 2:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_encoded,
            y=y_encoded
        )
        class_weight_dict = dict(zip(unique_encoded, class_weights))
        log_message(f"Using class weights: {class_weight_dict}")
        class_weights = class_weight_dict
    
    return X_std, y_categorical, unique_labels, class_weights

def create_3class_cnn_model(input_shape, num_classes):
    """Create CNN optimized for 3-class sleep apnea detection (no MixedApnea)"""
    log_message(f"Creating 3-class CNN model for input shape: {input_shape}")
    
    model = Sequential([
        # First convolutional block - capture rapid signal changes
        Conv1D(32, kernel_size=9, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv1D(32, kernel_size=15, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second convolutional block - medium-term patterns
        Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=11, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # Third convolutional block - long-term apnea patterns
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=21, activation='relu', padding='same'),  # ~0.1 second patterns
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Fourth block - global patterns
        Conv1D(256, kernel_size=11, activation='relu', padding='same'),
        BatchNormalization(),
        
        # Global average pooling
        GlobalAveragePooling1D(),
        
        # Dense layers for classification
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.2),
        
        # Output layer (3 classes: Normal, CentralApnea, ObstructiveApnea)
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer with moderate learning rate
    optimizer = Adam(
        learning_rate=0.0003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    log_message("3-class CNN model created successfully")
    log_message(f"Model parameters: {model.count_params():,}")
    return model

def balance_classes_strategic(X, y):
    """Strategic balancing for 3-class problem"""
    from sklearn.utils import resample
    
    log_message("Strategic balancing for 3-class classification...")
    
    # Get class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    min_count = min(counts)
    
    log_message(f"Original distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(y)) * 100
        log_message(f"  {label}: {count} ({percentage:.1f}%)")
    
    log_message(f"Class imbalance ratio: {max_count/min_count:.1f}:1")
    
    # For 3-class problem, aim for better balance
    X_balanced = []
    y_balanced = []
    
    # Set target counts to improve minority class representation
    # Use a conservative approach - don't oversample too aggressively
    median_count = int(np.median(counts))
    target_normal = min(max_count, median_count * 2)  # Limit normal class
    target_apnea = max(min_count * 2, 500)  # Boost apnea classes
    
    target_counts = {
        'Normal': target_normal,
        'CentralApnea': target_apnea,
        'ObstructiveApnea': target_apnea
    }
    
    log_message(f"Target distribution:")
    for label, target in target_counts.items():
        log_message(f"  {label}: {target}")
    
    for label in unique_labels:
        # Get samples for this class
        mask = (y == label)
        X_class = X[mask]
        y_class = y[mask]
        current_count = len(X_class)
        target_count = target_counts.get(label, current_count)
        
        if current_count < target_count:
            # Upsample
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=target_count,
                random_state=42,
                replace=True
            )
            log_message(f"  {label}: {current_count} -> {target_count} (upsampled)")
        elif current_count > target_count:
            # Downsample
            X_resampled, y_resampled = resample(
                X_class, y_class,
                n_samples=target_count,
                random_state=42,
                replace=False
            )
            log_message(f"  {label}: {current_count} -> {target_count} (downsampled)")
        else:
            X_resampled, y_resampled = X_class, y_class
            log_message(f"  {label}: {current_count} (kept)")
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    # Combine and shuffle
    X_final = np.vstack(X_balanced)
    y_final = np.hstack(y_balanced)
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    log_message(f"Final balanced data shape: {X_final.shape}")
    
    # Final distribution
    unique_final, counts_final = np.unique(y_final, return_counts=True)
    log_message(f"Final balanced distribution:")
    for label, count in zip(unique_final, counts_final):
        percentage = (count / len(y_final)) * 100
        log_message(f"  {label}: {count} ({percentage:.1f}%)")
    
    return X_final, y_final

def train_and_evaluate_cnn():
    """Main function to train and evaluate 3-class CNN"""
    log_file = CONFIG['output_file']
    
    if os.path.exists(log_file):
        os.remove(log_file)
    
    start_time = time.time()
    
    try:
        setup_cpu()
        log_message("🚀 STARTING 3-CLASS CNN TRAINING (NO MIXED APNEA)", log_file)
        log_message("=" * 70, log_file)
        
        # Load data with MixedApnea exclusion
        X, y = load_data_efficiently(
            CONFIG['data_path'], 
            CONFIG['channels'], 
            CONFIG['max_segments'],
            CONFIG['exclude_classes']
        )
        
        # Strategic balancing for 3-class problem
        X, y = balance_classes_strategic(X, y)
        
        # Preprocess
        X_processed, y_processed, label_names, class_weights = preprocess_data(X, y)
        
        log_message(f"Final processed data shape: {X_processed.shape}", log_file)
        log_message(f"Number of classes: {y_processed.shape[1]}", log_file)
        log_message(f"Classes: {', '.join(label_names)}", log_file)
        
        # Split data with stratification
        log_message("Splitting data into train/test sets...", log_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed,
            test_size=CONFIG['test_size'],
            random_state=CONFIG['random_state'],
            stratify=np.argmax(y_processed, axis=1)
        )
        
        log_message(f"Training set: {X_train.shape}", log_file)
        log_message(f"Test set: {X_test.shape}", log_file)
        
        # Create 3-class CNN model
        model = create_3class_cnn_model(X_train.shape[1:], y_processed.shape[1])
        
        # Callbacks with appropriate patience for 3-class problem
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=12,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        log_message("🏋️ Starting 3-class CNN training...", log_file)
        training_start = time.time()
        
        history = model.fit(
            X_train, y_train,
            epochs=150,
            batch_size=32,  # Moderate batch size
            validation_split=0.2,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        training_time = time.time() - training_start
        log_message(f"Training completed in: {timedelta(seconds=int(training_time))}", log_file)
        
        # Evaluate model
        log_message("📊 EVALUATING 3-CLASS MODEL", log_file)
        log_message("-" * 40, log_file)
        
        # Test set evaluation
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        log_message(f"Test Accuracy: {test_acc:.4f}", log_file)
        log_message(f"Test Loss: {test_loss:.4f}", log_file)
        
        # Predictions and detailed metrics
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Classification report
        log_message("\n3-Class Classification Report:", log_file)
        log_message("-" * 40, log_file)
        report = classification_report(
            y_test_classes, y_pred_classes,
            target_names=label_names,
            digits=4
        )
        for line in report.split('\n'):
            log_message(line, log_file)
        
        # Confusion matrix
        log_message("\nConfusion Matrix:", log_file)
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        log_message(f"True\\Pred  {' '.join([f'{label:>12s}' for label in label_names])}", log_file)
        for i, (label, row) in enumerate(zip(label_names, cm)):
            log_message(f"{label:>10s}  {' '.join([f'{val:>12d}' for val in row])}", log_file)
        
        # Training history summary
        log_message("\nTraining History Summary:", log_file)
        log_message("-" * 40, log_file)
        best_epoch = np.argmax(history.history['val_accuracy']) + 1
        best_val_acc = max(history.history['val_accuracy'])
        final_train_acc = history.history['accuracy'][-1]
        
        log_message(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})", log_file)
        log_message(f"Final training accuracy: {final_train_acc:.4f}", log_file)
        log_message(f"Total epochs trained: {len(history.history['accuracy'])}", log_file)
        
        # Calculate per-class performance
        log_message("\nPer-Class Performance Analysis:", log_file)
        log_message("-" * 40, log_file)
        
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_classes, y_pred_classes, average=None
        )
        
        for i, label in enumerate(label_names):
            log_message(f"{label}:", log_file)
            log_message(f"  Precision: {precision[i]:.4f}", log_file)
            log_message(f"  Recall: {recall[i]:.4f}", log_file)
            log_message(f"  F1-Score: {f1[i]:.4f}", log_file)
            log_message(f"  Support: {support[i]}", log_file)
        
        # Final summary
        total_time = time.time() - start_time
        log_message("\n" + "=" * 70, log_file)
        log_message("🎯 FINAL 3-CLASS RESULTS SUMMARY", log_file)
        log_message("=" * 70, log_file)
        log_message(f"Dataset: {len(X)} segments, {len(CONFIG['channels'])} channels", log_file)
        log_message(f"Classes: {', '.join(label_names)} (MixedApnea excluded)", log_file)
        log_message(f"Test Accuracy: {test_acc:.4f}", log_file)
        log_message(f"Best Validation Accuracy: {best_val_acc:.4f}", log_file)
        log_message(f"Training Time: {timedelta(seconds=int(training_time))}", log_file)
        log_message(f"Total Runtime: {timedelta(seconds=int(total_time))}", log_file)
        log_message("=" * 70, log_file)
        
        # Save model
        model_path = "cnn_model_3class.h5"
        model.save(model_path)
        log_message(f"Model saved to: {model_path}", log_file)
        
        log_message(f"✅ Results saved to: {log_file}")
        
    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}"
        log_message(error_msg, log_file)
        import traceback
        traceback.print_exc()
        
        # Log full traceback
        log_message("\nFull Error Traceback:", log_file)
        log_message("-" * 40, log_file)
        for line in traceback.format_exc().split('\n'):
            log_message(line, log_file)
    
    finally:
        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    train_and_evaluate_cnn()