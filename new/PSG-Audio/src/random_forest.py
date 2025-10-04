import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from collections import Counter
from config.config import CONFIG

def balance_classes_smart(X, y, strategy='smart'):
    """Smart balancing optimized for 285-patient dataset"""
    print(f"Smart balancing for large dataset (285 patients)...")
    
    # Get unique labels from categorical y
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    
    print("Original class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"  Class {label_idx}: {count} ({count/len(y)*100:.1f}%)")
    
    X_balanced = []
    y_balanced = []
    
    # Updated target counts for larger dataset
    target_counts = {
        0: 2000,   # CentralApnea - increased
        1: 8000,   # Normal - increased  
        2: 10000   # ObstructiveApnea - increased
    }
    
    for label_idx in unique_labels:
        mask = (y_labels == label_idx)
        X_class = X[mask]
        y_class = y[mask]
        current_count = len(X_class)
        target_count = target_counts.get(label_idx, current_count)
        
        if current_count < target_count:
            # Upsample
            indices = np.random.choice(len(X_class), target_count, replace=True)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            print(f"  Class {label_idx}: {current_count} -> {target_count} (upsampled)")
        elif current_count > target_count:
            # Downsample
            indices = np.random.choice(len(X_class), target_count, replace=False)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            print(f"  Class {label_idx}: {current_count} -> {target_count} (downsampled)")
        else:
            X_resampled, y_resampled = X_class, y_class
            print(f"  Class {label_idx}: {current_count} (kept)")
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    # Combine and shuffle
    X_final = np.vstack(X_balanced)
    y_final = np.vstack(y_balanced)
    
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    print(f"\nFinal balanced data shape: {X_final.shape}")
    
    # Final distribution
    y_final_labels = np.argmax(y_final, axis=1)
    unique_final, counts_final = np.unique(y_final_labels, return_counts=True)
    print("Final class distribution:")
    for label_idx, count in zip(unique_final, counts_final):
        print(f"  Class {label_idx}: {count} ({count/len(y_final)*100:.1f}%)")
    
    return X_final, y_final

def extract_advanced_features(channel_data):
    """Extract comprehensive statistical features"""
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
    features['skew'] = np.mean(((channel_data - features['mean']) / features['std'])**3) if features['std'] > 0 else 0
    features['kurtosis'] = np.mean(((channel_data - features['mean']) / features['std'])**4) if features['std'] > 0 else 0
    
    # Zero crossing rate
    features['zcr'] = np.sum(np.diff(np.sign(channel_data)) != 0) / len(channel_data)
    
    return features

def get_feature_importance(X, y, channels):
    """Get feature importance using optimized Random Forest"""
    print("🔍 EXTRACTING FEATURES FOR RANDOM FOREST (285-patient dataset)")
    print("=" * 70)
    
    # Extract advanced features for each channel
    feature_names = []
    X_features = []
    
    print("Extracting features from each channel...")
    for sample_idx in range(X.shape[0]):
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
        
        X_features.append(sample_features)
        
        if (sample_idx + 1) % 2000 == 0:  # Progress every 2000 samples for large dataset
            print(f"  Processed {sample_idx + 1}/{X.shape[0]} samples")
    
    X_features = np.array(X_features)
    print(f"Feature extraction complete: {X_features.shape}")
    print(f"Total features: {len(feature_names)}")
    
    # Balance classes
    X_balanced, y_balanced = balance_classes_smart(X_features, y)
    
    # Convert y to label indices for Random Forest
    y_rf = np.argmax(y_balanced, axis=1)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_rf),
        y=y_rf
    )
    class_weight_dict = dict(zip(np.unique(y_rf), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # Train Random Forest with updated config
    print("\n🌲 TRAINING RANDOM FOREST (Optimized for large dataset)")
    print("-" * 50)
    
    rf_config = CONFIG['rf_config'].copy()
    rf_config['class_weight'] = class_weight_dict
    
    print(f"RF Config: {rf_config}")
    
    rf = RandomForestClassifier(**rf_config)
    rf.fit(X_balanced, y_rf)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Enhanced feature importance analysis
    print(f"\n🔍 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    print(f"Total features extracted: {len(feature_importance)}")
    print(f"Sum of all importances: {sum(imp for _, imp in feature_importance):.6f}")
    print(f"Average importance: {np.mean([imp for _, imp in feature_importance]):.6f}")
    print(f"Top feature importance: {feature_importance[0][1]:.6f}")
    print(f"Bottom feature importance: {feature_importance[-1][1]:.6f}")
    print(f"Importance ratio (top/bottom): {feature_importance[0][1]/feature_importance[-1][1]:.1f}:1")
    
    # Expected theoretical uniform distribution
    uniform_importance = 1.0 / len(feature_importance)
    print(f"Uniform distribution would be: {uniform_importance:.6f}")
    print(f"Top feature vs uniform: {feature_importance[0][1]/uniform_importance:.1f}x higher")
    
    print("\n📊 TOP 20 FEATURE IMPORTANCES:")
    print("-" * 50)
    for i, (feature, importance) in enumerate(feature_importance[:20], 1):
        print(f"{i:2d}. {feature:30s}: {importance:.4f}")
    
    # True channel-level importance
    channel_importance = {}
    for feature_name, importance in feature_importance:
        # Extract channel name properly (handle compound names like "EEG A1-A2")
        parts = feature_name.split('_')
        if len(parts) >= 2:
            # Join all parts except the last one (which is the feature type)
            channel_name = '_'.join(parts[:-1])
            
            # Handle space-separated compound names
            if channel_name in channels:
                if channel_name not in channel_importance:
                    channel_importance[channel_name] = 0
                channel_importance[channel_name] += importance
    
    print(f"\n🔬 TRUE CHANNEL-LEVEL IMPORTANCE:")
    print("-" * 40)
    sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (channel, importance) in enumerate(sorted_channels, 1):
        print(f"{i:2d}. {channel:15s}: {importance:.4f}")
    
    # Verify we have all 9 channels
    print(f"\nChannel coverage: {len(channel_importance)}/9 channels")
    missing_channels = set(channels) - set(channel_importance.keys())
    if missing_channels:
        print(f"Missing channels: {missing_channels}")
    
    return feature_importance

def select_top_features(feature_importance, percentage=0.25):
    """Select top features by percentage"""
    n_features = int(len(feature_importance) * percentage)
    n_features = max(1, n_features)  # At least 1 feature
    
    top_features = [feat for feat, _ in feature_importance[:n_features]]
    
    print(f"\n📌 SELECTED TOP {percentage*100:.0f}% FEATURES ({len(top_features)} features):")
    print("-" * 50)
    for i, feature in enumerate(top_features, 1):
        importance = next(imp for feat, imp in feature_importance if feat == feature)
        print(f"{i:2d}. {feature:30s}: {importance:.4f}")
    
    return top_features

def convert_feature_names_to_channels(feature_names, channels):
    """Convert feature names back to channel indices for CNN"""
    channel_indices = []
    
    for feature_name in feature_names:
        # Extract channel name from feature name
        parts = feature_name.split('_')
        if len(parts) >= 2:
            channel_name = '_'.join(parts[:-1])
            
            if channel_name in channels:
                channel_idx = channels.index(channel_name)
                if channel_idx not in channel_indices:
                    channel_indices.append(channel_idx)
    
    # Convert back to channel names
    selected_channels = [channels[idx] for idx in sorted(channel_indices)]
    
    print(f"\n🎯 MAPPED TO CHANNELS ({len(selected_channels)} channels):")
    for i, channel in enumerate(selected_channels, 1):
        print(f"{i:2d}. {channel}")
    
    return selected_channels