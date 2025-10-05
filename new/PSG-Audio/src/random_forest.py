import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from collections import Counter
import psutil
import time
from config.config import CONFIG

def monitor_memory_usage():
    """Monitor memory usage if enabled"""
    if CONFIG['monitoring']['log_memory_usage']:
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.used/1e9:.1f}GB used / {memory.total/1e9:.1f}GB total ({memory.percent:.1f}%)")

def balance_classes_advanced(X, y, strategy='optimized'):
    """Advanced balancing optimized for 1TB RAM system"""
    print(f"🔄 ADVANCED CLASS BALANCING (1TB RAM Optimized)")
    print("-" * 50)
    
    # Get unique labels from categorical y
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    
    print("Original class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"  Class {label_idx}: {count:,} ({count/len(y)*100:.1f}%)")
    
    monitor_memory_usage()
    
    X_balanced = []
    y_balanced = []
    
    # Optimized target counts for large dataset with 1TB RAM
    target_counts = {
        0: 5000,   # CentralApnea - significantly increased
        1: 15000,  # Normal - increased for better representation  
        2: 20000   # ObstructiveApnea - increased for robust training
    }
    
    print(f"Target balanced counts: {target_counts}")
    
    for label_idx in unique_labels:
        mask = (y_labels == label_idx)
        X_class = X[mask]
        y_class = y[mask]
        current_count = len(X_class)
        target_count = target_counts.get(label_idx, current_count)
        
        if current_count < target_count:
            # Advanced upsampling with noise injection
            indices = np.random.choice(len(X_class), target_count, replace=True)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            
            # Add small amount of noise to prevent exact duplicates
            if CONFIG['training']['use_advanced_augmentation']:
                noise = np.random.normal(0, 0.01, X_resampled.shape)
                X_resampled = X_resampled + noise
            
            print(f"  Class {label_idx}: {current_count:,} -> {target_count:,} (upsampled with augmentation)")
        elif current_count > target_count:
            # Smart downsampling - keep diverse samples
            indices = np.random.choice(len(X_class), target_count, replace=False)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            print(f"  Class {label_idx}: {current_count:,} -> {target_count:,} (downsampled)")
        else:
            X_resampled, y_resampled = X_class, y_class
            print(f"  Class {label_idx}: {current_count:,} (kept)")
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    # Combine and shuffle with large buffer
    print("Combining and shuffling balanced data...")
    X_final = np.vstack(X_balanced)
    y_final = np.vstack(y_balanced)
    
    # Use large shuffle buffer for better randomization
    shuffle_buffer = min(len(X_final), CONFIG['data_loading']['shuffle_buffer_size'])
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    print(f"Final balanced data shape: {X_final.shape}")
    monitor_memory_usage()
    
    # Final distribution
    y_final_labels = np.argmax(y_final, axis=1)
    unique_final, counts_final = np.unique(y_final_labels, return_counts=True)
    print("Final balanced distribution:")
    for label_idx, count in zip(unique_final, counts_final):
        print(f"  Class {label_idx}: {count:,} ({count/len(y_final)*100:.1f}%)")
    
    return X_final, y_final

def extract_advanced_features(channel_data):
    """Extract comprehensive statistical and spectral features"""
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(channel_data)
    features['std'] = np.std(channel_data)
    features['var'] = np.var(channel_data)
    features['min'] = np.min(channel_data)
    features['max'] = np.max(channel_data)
    features['range'] = features['max'] - features['min']
    features['median'] = np.median(channel_data)
    features['rms'] = np.sqrt(np.mean(channel_data**2))
    
    # Percentiles
    features['p25'] = np.percentile(channel_data, 25)
    features['p75'] = np.percentile(channel_data, 75)
    features['iqr'] = features['p75'] - features['p25']
    
    # Higher-order statistics
    if features['std'] > 0:
        normalized = (channel_data - features['mean']) / features['std']
        features['skew'] = np.mean(normalized**3)
        features['kurtosis'] = np.mean(normalized**4)
    else:
        features['skew'] = 0
        features['kurtosis'] = 0
    
    # Signal characteristics
    features['zcr'] = np.sum(np.diff(np.sign(channel_data)) != 0) / len(channel_data)
    
    # Advanced features for 1TB RAM system
    if CONFIG['training']['use_advanced_augmentation']:
        # Energy and power features
        features['energy'] = np.sum(channel_data**2)
        features['power'] = features['energy'] / len(channel_data)
        
        # Entropy approximation
        hist, _ = np.histogram(channel_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        features['entropy'] = -np.sum(hist * np.log2(hist))
        
        # Spectral centroid approximation (simplified)
        fft = np.abs(np.fft.fft(channel_data))
        freqs = np.arange(len(fft))
        features['spectral_centroid'] = np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
    
    return features

def get_feature_importance(X, y, channels):
    """Get feature importance using optimized Random Forest for 1TB system"""
    print("🌲 OPTIMIZED RANDOM FOREST FEATURE EXTRACTION")
    print(f"System: 1TB RAM, {CONFIG['hardware']['n_processes']} cores")
    print("=" * 70)
    
    start_time = time.time()
    
    # Extract advanced features for each channel
    feature_names = []
    X_features = []
    
    print("Extracting advanced features from each channel...")
    monitor_memory_usage()
    
    # Process in optimized batches for 1TB RAM
    batch_size = 10000  # Much larger batches with ample RAM
    
    for batch_start in range(0, X.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X.shape[0])
        batch_features = []
        
        for sample_idx in range(batch_start, batch_end):
            sample_features = []
            
            for ch_idx, channel_name in enumerate(channels):
                channel_data = X[sample_idx, :, ch_idx]
                features = extract_advanced_features(channel_data)
                
                # Add to feature list (only once)
                if sample_idx == 0:
                    for feat_name, _ in features.items():
                        feature_names.append(f"{channel_name}_{feat_name}")
                
                # Add feature values
                sample_features.extend(features.values())
            
            batch_features.append(sample_features)
        
        X_features.extend(batch_features)
        
        if (batch_end) % 20000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {batch_end:,}/{X.shape[0]:,} samples ({elapsed:.1f}s)")
            monitor_memory_usage()
    
    X_features = np.array(X_features, dtype=np.float32)
    print(f"Feature extraction complete: {X_features.shape}")
    print(f"Total features: {len(feature_names)}")
    print(f"Feature extraction time: {time.time() - start_time:.1f}s")
    
    # Advanced class balancing
    X_balanced, y_balanced = balance_classes_advanced(X_features, y)
    
    # Convert y to label indices for Random Forest
    y_rf = np.argmax(y_balanced, axis=1)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_rf),
        y=y_rf
    )
    class_weight_dict = dict(zip(np.unique(y_rf), class_weights))
    print(f"Computed class weights: {class_weight_dict}")
    
    # Train optimized Random Forest
    print("\n🚀 TRAINING OPTIMIZED RANDOM FOREST")
    print("-" * 50)
    
    rf_config = CONFIG['rf_config'].copy()
    rf_config['class_weight'] = class_weight_dict
    
    print(f"RF Configuration:")
    for key, value in rf_config.items():
        print(f"  {key}: {value}")
    
    monitor_memory_usage()
    
    rf_start = time.time()
    rf = RandomForestClassifier(**rf_config)
    rf.fit(X_balanced, y_rf)
    rf_time = time.time() - rf_start
    
    print(f"Random Forest training completed in {rf_time:.1f}s")
    
    # Cross-validation if enabled
    if CONFIG['training']['cross_validation_folds'] > 1:
        print(f"\n🔄 CROSS-VALIDATION ({CONFIG['training']['cross_validation_folds']} folds)")
        cv_scores = cross_val_score(rf, X_balanced, y_rf, 
                                  cv=CONFIG['training']['cross_validation_folds'], 
                                  n_jobs=-1, verbose=1)
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Out-of-bag score
    if rf_config.get('oob_score', False):
        print(f"Out-of-bag score: {rf.oob_score_:.4f}")
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Enhanced analysis
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 50)
    print(f"Total features extracted: {len(feature_importance):,}")
    print(f"Sum of all importances: {sum(imp for _, imp in feature_importance):.6f}")
    print(f"Average importance: {np.mean([imp for _, imp in feature_importance]):.6f}")
    print(f"Top feature importance: {feature_importance[0][1]:.6f}")
    print(f"Bottom feature importance: {feature_importance[-1][1]:.6f}")
    print(f"Importance ratio (top/bottom): {feature_importance[0][1]/feature_importance[-1][1]:.1f}:1")
    
    uniform_importance = 1.0 / len(feature_importance)
    print(f"Uniform distribution: {uniform_importance:.6f}")
    print(f"Top feature vs uniform: {feature_importance[0][1]/uniform_importance:.1f}x higher")
    
    print("\n📋 TOP 25 FEATURE IMPORTANCES:")
    print("-" * 60)
    for i, (feature, importance) in enumerate(feature_importance[:25], 1):
        print(f"{i:2d}. {feature:35s}: {importance:.6f}")
    
    # True channel-level importance
    channel_importance = {}
    for feature_name, importance in feature_importance:
        parts = feature_name.split('_')
        if len(parts) >= 2:
            channel_name = '_'.join(parts[:-1])
            if channel_name in channels:
                if channel_name not in channel_importance:
                    channel_importance[channel_name] = 0
                channel_importance[channel_name] += importance
    
    print(f"\n🎯 CHANNEL-LEVEL IMPORTANCE RANKING:")
    print("-" * 50)
    sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (channel, importance) in enumerate(sorted_channels, 1):
        print(f"{i:2d}. {channel:20s}: {importance:.6f}")
    
    # Verify coverage
    print(f"\nChannel coverage: {len(channel_importance)}/{len(channels)} channels")
    missing_channels = set(channels) - set(channel_importance.keys())
    if missing_channels:
        print(f"⚠️  Missing channels: {missing_channels}")
    
    monitor_memory_usage()
    print(f"Total processing time: {time.time() - start_time:.1f}s")
    
    return feature_importance

def select_top_features(feature_importance, percentage=0.25):
    """Select top features by percentage with enhanced reporting"""
    n_features = int(len(feature_importance) * percentage)
    n_features = max(1, n_features)
    
    top_features = [feat for feat, _ in feature_importance[:n_features]]
    
    print(f"\n🎯 SELECTED TOP {percentage*100:.0f}% FEATURES ({len(top_features):,} features):")
    print("-" * 60)
    
    # Show top features with their importance and rank
    for i, feature in enumerate(top_features[:20], 1):  # Show top 20
        importance = next(imp for feat, imp in feature_importance if feat == feature)
        print(f"{i:2d}. {feature:35s}: {importance:.6f}")
    
    if len(top_features) > 20:
        print(f"... and {len(top_features) - 20:,} more features")
    
    return top_features

def convert_feature_names_to_channels(feature_names, channels):
    """Convert feature names back to channel indices for CNN"""
    channel_indices = []
    
    for feature_name in feature_names:
        parts = feature_name.split('_')
        if len(parts) >= 2:
            channel_name = '_'.join(parts[:-1])
            
            if channel_name in channels:
                channel_idx = channels.index(channel_name)
                if channel_idx not in channel_indices:
                    channel_indices.append(channel_idx)
    
    selected_channels = [channels[idx] for idx in sorted(channel_indices)]
    
    print(f"\n📍 MAPPED TO CHANNELS ({len(selected_channels)} channels):")
    for i, channel in enumerate(selected_channels, 1):
        print(f"{i:2d}. {channel}")
    
    return selected_channels