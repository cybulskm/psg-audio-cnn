import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy import stats
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def log_memory_usage():
    """Log current memory usage"""
    import psutil
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB ({memory_mb/1024:.1f} GB)")

def extract_advanced_features(channel_data):
    """Extract advanced statistical features from time series data"""
    if len(channel_data) == 0:
        return [0.0] * 12  # Return zeros for all features
    
    data = np.array(channel_data)
    
    # Remove NaN values
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return [0.0] * 12
    
    features = []
    
    # Basic statistics
    features.append(np.mean(data))           # Mean
    features.append(np.std(data))            # Standard deviation
    features.append(np.var(data))            # Variance
    features.append(stats.skew(data))        # Skewness
    features.append(stats.kurtosis(data))    # Kurtosis
    
    # Percentiles
    features.append(np.percentile(data, 25)) # 25th percentile
    features.append(np.percentile(data, 75)) # 75th percentile
    features.append(np.max(data))            # Maximum
    features.append(np.min(data))            # Minimum
    
    # Signal characteristics
    features.append(np.ptp(data))            # Peak-to-peak (range)
    
    # Zero crossing rate (for detecting signal changes)
    zero_crossings = len(np.where(np.diff(np.signbit(data)))[0])
    features.append(zero_crossings / len(data))
    
    # RMS (Root Mean Square) - energy measure
    features.append(np.sqrt(np.mean(data**2)))
    
    return features

def get_channels_from_pickle(data_path):
    """Extract channel names from pickle file"""
    print("Extracting channels from pickle file...")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected non-empty list of segments")
    
    # Get first valid segment
    first_segment = None
    for seg in data:
        if isinstance(seg, dict) and 'Label' in seg:
            first_segment = seg
            break
    
    if first_segment is None:
        raise ValueError("No valid segments found with Label")
    
    # Extract all keys except 'Label'
    channels = [key for key in first_segment.keys() if key != 'Label']
    
    print(f"Found {len(channels)} channels: {channels}")
    
    return channels

def load_data_with_advanced_features(data_path, channels=None, max_segments=None, batch_size=500):
    """Load data with advanced feature extraction"""
    print("Loading data with advanced feature extraction...")
    log_memory_usage()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected list of segments")
    
    # Auto-detect channels if not provided
    if channels is None:
        channels = get_channels_from_pickle(data_path)
    
    total_segments = len(data)
    print(f"Total segments in file: {total_segments}")
    print(f"Using channels: {channels}")
    
    if max_segments and total_segments > max_segments:
        print(f"Using first {max_segments} segments for memory management")
        data = data[:max_segments]
        total_segments = max_segments
    
    X_list = []
    y_list = []
    valid_count = 0
    invalid_count = 0
    
    # Process in batches
    for batch_start in range(0, total_segments, batch_size):
        batch_end = min(batch_start + batch_size, total_segments)
        batch = data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}: segments {batch_start}-{batch_end}")
        
        for seg in batch:
            if not isinstance(seg, dict) or 'Label' not in seg:
                invalid_count += 1
                continue
            
            # Extract advanced features for each channel
            all_features = []
            valid_segment = True
            
            for ch in channels:
                if ch in seg and seg[ch] is not None:
                    ch_data = seg[ch]
                    if isinstance(ch_data, (list, np.ndarray)) and len(ch_data) > 0:
                        # Extract 12 features per channel
                        ch_features = extract_advanced_features(ch_data)
                        all_features.extend(ch_features)
                    else:
                        all_features.extend([0.0] * 12)
                else:
                    valid_segment = False
                    break
            
            if valid_segment and len(all_features) == len(channels) * 12:
                X_list.append(all_features)
                y_list.append(seg['Label'])
                valid_count += 1
            else:
                invalid_count += 1
        
        # Clear batch
        del batch
        gc.collect()
        
        if (batch_start // batch_size + 1) % 5 == 0:
            print(f"  Processed {batch_end} segments - Valid: {valid_count}, Invalid: {invalid_count}")
            log_memory_usage()
    
    del data
    gc.collect()
    
    print(f"Final processing results - Valid: {valid_count}, Invalid: {invalid_count}")
    
    if not X_list:
        raise ValueError("No valid segments found!")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    
    del X_list, y_list
    gc.collect()
    
    print(f"Final data shape: {X.shape} ({len(channels)} channels × 12 features = {X.shape[1]} total features)")
    print(f"Memory usage: {X.nbytes / 1e6:.1f} MB")
    
    return X, y, channels

def balance_classes_advanced(X, y, strategy='hybrid'):
    """Advanced class balancing with multiple strategies"""
    print(f"Balancing classes using {strategy} strategy...")
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Original class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio <= 3:
        print("Classes are reasonably balanced, no resampling needed.")
        return X, y
    
    if strategy == 'hybrid':
        # Hybrid approach: moderate upsampling + downsampling
        target_min_samples = max(200, min_count * 2)  # At least 200 samples per class
        target_max_samples = min(2000, max_count // 2)  # Cap majority class
        
        X_balanced = []
        y_balanced = []
        
        for label in unique_labels:
            mask = (y == label)
            X_class = X[mask]
            y_class = y[mask]
            current_count = len(X_class)
            
            if current_count < target_min_samples:
                # Upsample minority classes
                X_resampled, y_resampled = resample(
                    X_class, y_class,
                    n_samples=target_min_samples,
                    random_state=42,
                    replace=True
                )
                print(f"  {label}: {current_count} -> {target_min_samples} (upsampled)")
            elif current_count > target_max_samples:
                # Downsample majority classes
                X_resampled, y_resampled = resample(
                    X_class, y_class,
                    n_samples=target_max_samples,
                    random_state=42,
                    replace=False
                )
                print(f"  {label}: {current_count} -> {target_max_samples} (downsampled)")
            else:
                X_resampled, y_resampled = X_class, y_class
                print(f"  {label}: {current_count} (kept)")
            
            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)
        
        X_final = np.vstack(X_balanced)
        y_final = np.hstack(y_balanced)
        
        # Shuffle
        indices = np.random.permutation(len(X_final))
        X_final = X_final[indices]
        y_final = y_final[indices]
        
        print(f"Balanced data shape: {X_final.shape}")
        unique_final, counts_final = np.unique(y_final, return_counts=True)
        print("Final class distribution:")
        for label, count in zip(unique_final, counts_final):
            print(f"  {label}: {count} ({count/len(y_final)*100:.1f}%)")
        
        return X_final, y_final
    
    return X, y

def optimize_random_forest(X_train, y_train):
    """Optimize Random Forest hyperparameters"""
    print("Optimizing Random Forest hyperparameters...")
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Grid search with key parameters
    param_grid = {
        'n_estimators': [150, 200],
        'max_depth': [15, 20, 25],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    rf_base = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Use balanced class weights
    )
    
    # Use smaller subset for grid search to save time
    if len(X_train) > 3000:
        indices = np.random.choice(len(X_train), 3000, replace=False)
        X_search = X_train[indices]
        y_search = y_train[indices]
    else:
        X_search = X_train
        y_search = y_train
    
    print(f"Running grid search on {len(X_search)} samples...")
    
    grid_search = GridSearchCV(
        rf_base,
        param_grid,
        cv=3,  # 3-fold CV to save time
        scoring='f1_macro',  # Better for imbalanced classes
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_search, y_search)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def create_feature_names(channels):
    """Create feature names for all extracted features"""
    feature_types = ['mean', 'std', 'var', 'skew', 'kurtosis', 
                    'p25', 'p75', 'max', 'min', 'range', 'zcr', 'rms']
    
    feature_names = []
    for channel in channels:
        for feat_type in feature_types:
            feature_names.append(f"{channel}_{feat_type}")
    
    return feature_names

def diagnose_pickle_data(data_path):
    """Diagnose what's actually in the pickle file"""
    print("🔍 DIAGNOSING PICKLE FILE CONTENTS")
    print("="*50)
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"Total segments: {len(data)}")
    
    # Count labels
    label_counts = {}
    apnea_total = 0
    
    for seg in data:
        if isinstance(seg, dict) and 'Label' in seg:
            label = seg['Label']
            label_counts[label] = label_counts.get(label, 0) + 1
            
            # Count total apnea events
            if 'apnea' in label.lower():
                apnea_total += 1
    
    print("Raw label distribution in pickle file:")
    total = sum(label_counts.values())
    normal_count = label_counts.get('Normal', 0)
    
    for label, count in sorted(label_counts.items()):
        percentage = (count / total) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nPreprocessing balance check:")
    print(f"  Normal events: {normal_count:,}")
    print(f"  Total apnea events: {apnea_total:,}")
    print(f"  Normal/Apnea ratio: {normal_count/apnea_total:.2f}:1" if apnea_total > 0 else "  No apnea events!")
    
    # Check imbalance among apnea subtypes
    apnea_labels = {k: v for k, v in label_counts.items() if 'apnea' in k.lower()}
    if len(apnea_labels) > 1:
        max_apnea = max(apnea_labels.values())
        min_apnea = min(apnea_labels.values())
        print(f"  Apnea subtype imbalance: {max_apnea/min_apnea:.1f}:1")

def balance_classes_smart(X, y, strategy='smart'):
    """Smart class balancing that works with preprocessed data"""
    print(f"Smart balancing for preprocessed data...")
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Original class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Separate normal and apnea classes
    normal_count = 0
    apnea_counts = {}
    
    for label, count in zip(unique_labels, counts):
        if label == 'Normal':
            normal_count = count
        elif 'apnea' in label.lower():
            apnea_counts[label] = count
    
    total_apnea = sum(apnea_counts.values())
    
    print(f"\nPreprocessing verification:")
    print(f"  Normal: {normal_count}")
    print(f"  Total Apnea: {total_apnea}")
    print(f"  Ratio: {normal_count/total_apnea:.2f}:1" if total_apnea > 0 else "No apnea!")
    
    # Check if we need apnea subtype balancing
    if len(apnea_counts) > 1:
        max_apnea = max(apnea_counts.values())
        min_apnea = min(apnea_counts.values())
        apnea_imbalance = max_apnea / min_apnea
        
        print(f"  Apnea subtype imbalance: {apnea_imbalance:.1f}:1")
        
        if apnea_imbalance > 5:  # Only balance if severe imbalance
            print("Balancing apnea subtypes...")
            
            # Target: balance apnea subtypes while keeping normal/apnea ratio
            target_apnea_per_type = max(100, min_apnea * 2)  # At least 100 samples
            
            X_balanced = []
            y_balanced = []
            
            for label in unique_labels:
                mask = (y == label)
                X_class = X[mask]
                y_class = y[mask]
                current_count = len(X_class)
                
                if label == 'Normal':
                    # Keep normal as is or downsample slightly if too large
                    if current_count > total_apnea * 2:
                        target_normal = total_apnea * 2
                        X_resampled, y_resampled = resample(
                            X_class, y_class,
                            n_samples=target_normal,
                            random_state=42,
                            replace=False
                        )
                        print(f"  {label}: {current_count} -> {target_normal} (downsampled)")
                    else:
                        X_resampled, y_resampled = X_class, y_class
                        print(f"  {label}: {current_count} (kept)")
                
                elif 'apnea' in label.lower() and current_count < target_apnea_per_type:
                    # Upsample minority apnea classes
                    X_resampled, y_resampled = resample(
                        X_class, y_class,
                        n_samples=target_apnea_per_type,
                        random_state=42,
                        replace=True
                    )
                    print(f"  {label}: {current_count} -> {target_apnea_per_type} (upsampled)")
                
                else:
                    # Keep as is
                    X_resampled, y_resampled = X_class, y_class
                    print(f"  {label}: {current_count} (kept)")
                
                X_balanced.append(X_resampled)
                y_balanced.append(y_resampled)
            
            # Combine and shuffle
            X_final = np.vstack(X_balanced)
            y_final = np.hstack(y_balanced)
            
            indices = np.random.permutation(len(X_final))
            X_final = X_final[indices]
            y_final = y_final[indices]
            
            print(f"\nFinal balanced data shape: {X_final.shape}")
            unique_final, counts_final = np.unique(y_final, return_counts=True)
            print("Final class distribution:")
            for label, count in zip(unique_final, counts_final):
                print(f"  {label}: {count} ({count/len(y_final)*100:.1f}%)")
            
            return X_final, y_final
        else:
            print("Apnea subtypes are reasonably balanced, keeping as is.")
    
    return X, y

def create_ensemble_for_mixed_apnea(X_train, y_train):
    """Create specialized ensemble to handle MixedApnea better"""
    print("Creating MixedApnea-focused ensemble...")
    
    # Create binary classifiers for better separation
    models = {}
    
    # 1. Mixed vs Other Apnea classifier
    y_mixed_binary = np.where(y_train == 1, 1, 0)  # 1 = MixedApnea, 0 = Other
    
    rf_mixed = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_mixed.fit(X_train, y_mixed_binary)
    models['mixed_detector'] = rf_mixed
    
    # 2. Main multi-class classifier
    rf_main = RandomForestClassifier(
        n_estimators=250,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='log2',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_main.fit(X_train, y_train)
    models['main_classifier'] = rf_main
    
    return models

def predict_with_ensemble(models, X_test):
    """Make predictions using ensemble approach"""
    # Get mixed apnea probability
    mixed_proba = models['mixed_detector'].predict_proba(X_test)[:, 1]
    main_pred = models['main_classifier'].predict(X_test)
    
    # Boost MixedApnea predictions where confidence is high
    final_pred = main_pred.copy()
    mixed_boost_threshold = 0.6
    
    for i in range(len(final_pred)):
        if mixed_proba[i] > mixed_boost_threshold:
            final_pred[i] = 1  # Force MixedApnea prediction
    
    return final_pred

def train_improved_random_forest():
    """Main function that works with preprocessed balanced data"""
    start_time = datetime.now()
    print("="*80)
    print("IMPROVED RANDOM FOREST FOR PREPROCESSED DATA")
    print("="*80)
    
    # Configuration
    data_path = "/raid/userdata/cybulskm/ThesisProj/285_patients_processed_means.pkl"
    max_segments = 15000
    
    try:
        # First diagnose what's in the file
        diagnose_pickle_data(data_path)
        print("\n" + "="*50 + "\n")
        
        # Load data with advanced features (channels auto-detected)
        X, y, channels = load_data_with_advanced_features(data_path, channels=None, max_segments=max_segments)
        
        # Clean data
        print("Cleaning data...")
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"Removing {nan_mask.sum()} rows with NaN values")
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        # Smart balancing (only balance apnea subtypes if needed)
        X, y = balance_classes_smart(X, y, strategy='smart')
        
        # Encode labels
        labels, y_encoded = np.unique(y, return_inverse=True)
        print(f"Found {len(labels)} classes: {labels}")
        
        # Feature scaling (helps with some RF implementations)
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        log_memory_usage()
        
        # Optimize Random Forest
        training_start = datetime.now()
        best_rf = optimize_random_forest(X_train, y_train)
        
        # Train final model on full training set
        print("Training final optimized model...")
        best_rf.fit(X_train, y_train)
        training_time = datetime.now() - training_start
        
        print(f"Training completed in: {training_time}")
        log_memory_usage()
        
        # Evaluate
        print("Evaluating model...")
        y_pred = best_rf.predict(X_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_rf, X_train, y_train, cv=3, scoring='f1_macro')
        print(f"Cross-validation F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Detailed results
        report = classification_report(y_test, y_pred, target_names=labels, digits=4)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance with names
        feature_names = create_feature_names(channels)
        importances = best_rf.feature_importances_
        feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        
        # Display results
        print("\nClassification Report:")
        print(report)
        
        print("\nConfusion Matrix:")
        print("True\\Pred", "\t".join([f"{label:>12s}" for label in labels]))
        for i, (label, row) in enumerate(zip(labels, cm)):
            print(f"{label:>8s}", "\t".join([f"{val:>12d}" for val in row]))
        
        print(f"\nTop 20 Feature Importances:")
        for i, (feature, importance) in enumerate(feature_importance[:20]):
            print(f"{i+1:2d}. {feature:25s}: {importance:.4f}")
        
        # Channel-level importance (sum across all features per channel)
        channel_importance = {}
        for channel in channels:
            channel_sum = sum(imp for feat, imp in feature_importance if feat.startswith(channel))
            channel_importance[channel] = channel_sum
        
        sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
        print(f"\nChannel-Level Importance (sum of all features per channel):")
        for channel, importance in sorted_channels:
            print(f"{channel:15s}: {importance:.4f}")
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_path = os.path.join(results_dir, f"improved_rf_results_{timestamp}.txt")
        
        with open(results_path, "w") as f:
            f.write("IMPROVED RANDOM FOREST WITH ADVANCED FEATURES\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total runtime: {datetime.now() - start_time}\n")
            f.write(f"Training time: {training_time}\n")
            f.write(f"Dataset size: {len(X)} segments\n")
            f.write(f"Channels: {channels}\n")
            f.write(f"Features: {X.shape[1]} advanced features ({len(channels)} channels × 12 features)\n")
            f.write(f"Max segments used: {max_segments}\n\n")
            
            f.write(f"Best hyperparameters: {best_rf.get_params()}\n\n")
            f.write(f"Cross-validation F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
            
            f.write("Final Class Distribution:\n")
            unique_final, counts_final = np.unique(y_encoded, return_counts=True)
            for label, count in zip(labels, counts_final):
                f.write(f"  {label}: {count} ({count/len(y_encoded)*100:.1f}%)\n")
            f.write("\n")
            
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nTop 20 Feature Importances:\n")
            for i, (feature, importance) in enumerate(feature_importance[:20]):
                f.write(f"{i+1:2d}. {feature:25s}: {importance:.4f}\n")
            
            f.write(f"\nChannel-Level Importance:\n")
            for channel, importance in sorted_channels:
                f.write(f"{channel:15s}: {importance:.4f}\n")
        
        print(f"\n✅ Comprehensive results saved to: {results_path}")
        total_runtime = datetime.now() - start_time
        print(f"Total runtime: {total_runtime}")
        
        # Expected improvements summary
        print(f"\n🎯 Expected Improvements vs Original:")
        print(f"• Channels auto-detected from pickle file: {len(channels)} channels")
        print(f"• More features: 9 → {X.shape[1]} features")
        print(f"• Better class balance through hybrid resampling")
        print(f"• Optimized hyperparameters via grid search")
        print(f"• Class-weighted learning")
        print(f"• Advanced statistical features capturing temporal patterns")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gc.collect()
        log_memory_usage()

if __name__ == "__main__":
    train_improved_random_forest()

