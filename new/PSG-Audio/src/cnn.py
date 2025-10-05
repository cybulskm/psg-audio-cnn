import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import psutil
import time
import os
from config.config import CONFIG

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

def monitor_memory_usage():
    """Monitor memory usage if enabled"""
    if CONFIG['monitoring']['log_memory_usage']:
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.used/1e9:.1f}GB used / {memory.total/1e9:.1f}GB total ({memory.percent:.1f}%)")

def preprocess_data_for_cnn(X, y):
    """Advanced preprocessing for CNN with 1TB RAM optimization"""
    print("🔧 ADVANCED CNN PREPROCESSING (1TB RAM System)")
    print("-" * 50)
    
    monitor_memory_usage()
    
    # Standardize the data with optimized processing
    print("Standardizing data with advanced normalization...")
    X_std = np.copy(X).astype(np.float32)
    
    for i in range(X.shape[-1]):
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
        
        print(f"  Channel {i}: mean={mean:.3f}, std={std:.3f}, "
              f"range=[{channel_data.min():.3f}, {channel_data.max():.3f}]")
    
    # Advanced class imbalance handling
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    imbalance_ratio = max(counts) / min(counts)
    
    print(f"\nClass distribution analysis:")
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count:,} samples ({count/len(y)*100:.1f}%)")
    
    class_weights = None
    if imbalance_ratio > 2:
        print(f"Computing advanced class weights for imbalance ratio: {imbalance_ratio:.1f}")
        
        # Enhanced class weight computation
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=y_labels
        )
        
        # Smooth extreme weights for stability
        max_weight = 10.0  # Cap weights to prevent instability
        class_weights = np.minimum(class_weights, max_weight)
        
        class_weight_dict = dict(zip(unique_labels, class_weights))
        print(f"Computed class weights: {class_weight_dict}")
        class_weights = class_weight_dict
    
    monitor_memory_usage()
    return X_std, y, class_weights

def create_advanced_cnn(input_shape, num_classes):
    """Create advanced CNN optimized for 1TB RAM system"""
    print(f"🏗️  BUILDING ADVANCED CNN ARCHITECTURE")
    print(f"Input shape: {input_shape}")
    print(f"Target classes: {num_classes}")
    print(f"Optimization level: 1TB RAM System")
    
    cnn_config = CONFIG['cnn_config']
    dropout_rates = cnn_config['dropout_rates']
    weight_decay = cnn_config.get('weight_decay', 1e-4)
    
    print(f"Dropout rates: {dropout_rates}")
    print(f"Weight decay: {weight_decay}")
    
    model = Sequential([
        # First convolutional block - fine-grained feature detection
        Conv1D(64, kernel_size=7, activation='relu', input_shape=input_shape, 
               padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv1D(64, kernel_size=11, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[0]),
        
        # Second block - medium-term patterns
        Conv1D(128, kernel_size=9, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv1D(128, kernel_size=15, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[1]),
        
        # Third block - long-term apnea patterns
        Conv1D(256, kernel_size=7, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv1D(256, kernel_size=21, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[2]),
        
        # Fourth block - global context
        Conv1D(512, kernel_size=11, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Conv1D(512, kernel_size=31, activation='relu', padding='same',
               kernel_regularizer=l2(weight_decay)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[3]),
        
        # Global feature extraction
        GlobalAveragePooling1D(),
        
        # Enhanced dense layers
        Dense(1024, activation='relu', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Dropout(dropout_rates[4]),
        
        Dense(512, activation='relu', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Dropout(dropout_rates[5]),
        
        Dense(256, activation='relu', kernel_regularizer=l2(weight_decay)),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Advanced optimizer selection
    optimizer_type = cnn_config.get('use_advanced_optimizer', 'adam')
    
    if optimizer_type == 'adamw':
        optimizer = AdamW(
            learning_rate=cnn_config['learning_rate'],
            weight_decay=cnn_config.get('weight_decay', 1e-4),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        print("Using AdamW optimizer with weight decay")
    else:
        optimizer = Adam(
            learning_rate=cnn_config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        print("Using Adam optimizer")
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Advanced CNN created successfully")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([np.prod(layer.get_weights()[0].shape) for layer in model.layers if layer.get_weights()])}")
    
    return model

def setup_advanced_callbacks(feature_description=""):
    """Setup advanced callbacks for training monitoring"""
    cnn_config = CONFIG['cnn_config']
    callbacks = []
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=cnn_config.get('early_stopping_metric', 'val_loss'),
        patience=cnn_config['patience'],
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Learning rate reduction
    lr_reducer = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=cnn_config['reduce_lr_patience'],
        min_lr=cnn_config['min_lr'],
        verbose=1
    )
    callbacks.append(lr_reducer)
    
    # Model checkpointing if enabled
    if CONFIG['training']['save_checkpoints']:
        checkpoint_dir = os.path.join(CONFIG['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'model_{feature_description}_{{epoch:02d}}.keras')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_freq=CONFIG['training']['checkpoint_frequency'],
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # TensorBoard logging if enabled
    if CONFIG['monitoring']['tensorboard_logging']:
        log_dir = os.path.join(CONFIG['output_dir'], 'tensorboard', feature_description)
        os.makedirs(log_dir, exist_ok=True)
        
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            profile_batch=0  # Disable profiling for CPU-only
        )
        callbacks.append(tensorboard)
    
    return callbacks

def train_and_evaluate_cnn(X, y, feature_description=""):
    """Advanced CNN training and evaluation with full optimization"""
    print(f"\n🚀 ADVANCED CNN TRAINING {feature_description}")
    print(f"System: 1TB RAM, CPU-only, {CONFIG['hardware']['n_processes']} cores")
    print("=" * 80)
    
    start_time = time.time()
    
    # Advanced preprocessing
    X_processed, y_processed, class_weights = preprocess_data_for_cnn(X, y)
    
    print(f"\nDataset information:")
    print(f"  Input shape: {X_processed.shape}")
    print(f"  Output shape: {y_processed.shape}")
    print(f"  Memory usage: {X_processed.nbytes / 1e9:.2f} GB")
    
    # Optimized data splitting with large batch consideration
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=np.argmax(y_processed, axis=1)
    )
    
    print(f"\nData splits:")
    print(f"  Training set: {X_train.shape} ({X_train.nbytes / 1e9:.2f} GB)")
    print(f"  Test set: {X_test.shape} ({X_test.nbytes / 1e9:.2f} GB)")
    
    monitor_memory_usage()
    
    # Create advanced model
    model = create_advanced_cnn(X_train.shape[1:], y_processed.shape[1])
    
    # Setup callbacks
    callbacks = setup_advanced_callbacks(feature_description.replace(' ', '_').replace('-', '_'))
    
    # Training configuration
    cnn_config = CONFIG['cnn_config']
    
    print(f"\n🏋️  STARTING ADVANCED TRAINING")
    print("-" * 50)
    print(f"Configuration:")
    print(f"  Epochs: {cnn_config['epochs']}")
    print(f"  Batch size: {cnn_config['batch_size']}")
    print(f"  Learning rate: {cnn_config['learning_rate']}")
    print(f"  Validation split: {CONFIG['validation_split']}")
    print(f"  Callbacks: {len(callbacks)}")
    
    # Train model with optimized settings
    history = model.fit(
        X_train, y_train,
        epochs=cnn_config['epochs'],
        batch_size=cnn_config['batch_size'],
        validation_split=CONFIG['validation_split'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        shuffle=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} minutes)")
    
    # Advanced evaluation
    print("\n📊 ADVANCED MODEL EVALUATION")
    print("-" * 50)
    
    # Test evaluation
    eval_start = time.time()
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    eval_time = time.time() - eval_start
    
    print(f"Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Evaluation time: {eval_time:.2f}s")
    
    # Detailed predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Enhanced classification report
    label_names = ['CentralApnea', 'Normal', 'ObstructiveApnea']
    print(f"\nDetailed Classification Report {feature_description}:")
    print("-" * 70)
    
    report = classification_report(
        y_test_classes, y_pred_classes,
        target_names=label_names,
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Enhanced confusion matrix
    print(f"\nConfusion Matrix {feature_description}:")
    print("-" * 50)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Header
    print(f"{'True\\Pred':<12} {' '.join([f'{label:>12s}' for label in label_names])}")
    print("-" * (12 + 13 * len(label_names)))
    
    # Matrix with percentages
    for i, (label, row) in enumerate(zip(label_names, cm)):
        row_total = row.sum()
        row_str = f"{label:<12}"
        for val in row:
            percentage = (val / row_total * 100) if row_total > 0 else 0
            row_str += f" {val:>8d}({percentage:>2.0f}%)"
        print(row_str)
    
    # Training history analysis
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_loss = min(history.history['val_loss'])
    
    print(f"\nTraining Summary {feature_description}:")
    print("-" * 50)
    print(f"  Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  Final training accuracy: {final_train_acc:.4f}")
    print(f"  Best validation loss: {final_val_loss:.4f}")
    print(f"  Total epochs trained: {len(history.history['accuracy'])}")
    print(f"  Training time: {training_time:.1f}s")
    print(f"  Time per epoch: {training_time/len(history.history['accuracy']):.1f}s")
    
    # Memory usage summary
    monitor_memory_usage()
    
    return test_acc, model, history