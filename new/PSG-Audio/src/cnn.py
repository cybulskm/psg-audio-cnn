import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config.config import CONFIG
import tensorflow as tf

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

def preprocess_data_for_cnn(X, y):
    """Preprocess data for CNN (same as your original)"""
    print("📊 PREPROCESSING FOR CNN:")
    print("-" * 30)
    
    # Standardize the data (critical for CNNs)
    print("Standardizing data...")
    X_std = np.copy(X)
    
    for i in range(X.shape[-1]):  # For each channel
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
        print(f"  Channel {i}: mean={mean:.3f}, std={std:.3f}")
    
    # Check class distribution and compute weights if needed
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    imbalance_ratio = max(counts) / min(counts)
    
    class_weights = None
    if imbalance_ratio > 2:
        print(f"Computing class weights for imbalance ratio: {imbalance_ratio:.1f}")
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_labels,
            y=y_labels
        )
        class_weight_dict = dict(zip(unique_labels, class_weights))
        print(f"Class weights: {class_weight_dict}")
        class_weights = class_weight_dict
    
    return X_std, y, class_weights

def create_improved_cnn(input_shape, num_classes):
    """Enhanced CNN for sleep apnea detection (same as your original)"""
    print(f"Creating CNN model for input shape: {input_shape}")
    
    cnn_config = CONFIG['cnn_config']
    dropout_rates = cnn_config['dropout_rates']
    
    model = Sequential([
        # First convolutional block - capture rapid signal changes
        Conv1D(32, kernel_size=9, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        Conv1D(32, kernel_size=15, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[0]),
        
        # Second convolutional block - medium-term patterns
        Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=11, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[1]),
        
        # Third convolutional block - long-term apnea patterns
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=21, activation='relu', padding='same'),  # ~0.1 second patterns
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rates[2]),
        
        # Fourth block - global patterns
        Conv1D(256, kernel_size=11, activation='relu', padding='same'),
        BatchNormalization(),
        
        # Global average pooling
        GlobalAveragePooling1D(),
        
        # Dense layers for classification
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rates[3]),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rates[4]),
        
        Dense(128, activation='relu'),
        Dropout(dropout_rates[5]),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer with same config as original
    optimizer = Adam(
        learning_rate=cnn_config['learning_rate'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("CNN model created successfully")
    print(f"Model parameters: {model.count_params():,}")
    return model

def train_and_evaluate_cnn(X, y, feature_description=""):
    """Train and evaluate CNN (same logic as your original)"""
    print(f"\n🚀 TRAINING CNN {feature_description}")
    print("=" * 60)
    
    # Preprocess data
    X_processed, y_processed, class_weights = preprocess_data_for_cnn(X, y)
    
    print(f"Input shape: {X_processed.shape}")
    print(f"Output shape: {y_processed.shape}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=np.argmax(y_processed, axis=1)
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create model
    model = create_improved_cnn(X_train.shape[1:], y_processed.shape[1])
    
    # Callbacks (same as your original)
    cnn_config = CONFIG['cnn_config']
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=cnn_config['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=cnn_config['reduce_lr_patience'],
            min_lr=cnn_config['min_lr'],
            verbose=1
        )
    ]
    
    # Train model
    print("🏋️ Starting CNN training...")
    history = model.fit(
        X_train, y_train,
        epochs=cnn_config['epochs'],
        batch_size=cnn_config['batch_size'],
        validation_split=CONFIG['validation_split'],
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    print("📊 EVALUATING CNN")
    print("-" * 40)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    label_names = ['CentralApnea', 'Normal', 'ObstructiveApnea']  # Assuming 3-class
    print(f"\nClassification Report {feature_description}:")
    print("-" * 40)
    report = classification_report(
        y_test_classes, y_pred_classes,
        target_names=label_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    print(f"\nConfusion Matrix {feature_description}:")
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    print(f"True\\Pred  {' '.join([f'{label:>12s}' for label in label_names])}")
    for i, (label, row) in enumerate(zip(label_names, cm)):
        print(f"{label:>10s}  {' '.join([f'{val:>12d}' for val in row])}")
    
    # Training history summary
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    final_train_acc = history.history['accuracy'][-1]
    
    print(f"\nTraining Summary {feature_description}:")
    print(f"Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Total epochs trained: {len(history.history['accuracy'])}")
    
    return test_acc, model, history