import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from collections import Counter
from config.config import CONFIG

def balance_classes_smart(X, y, strategy='smart'):
    """Smart balancing with better MixedApnea handling (same as your original)"""
    print(f"Smart balancing...")
    
    # Get unique labels from categorical y
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    
    print("Original class distribution:")
    for i, (label_idx, count) in enumerate(zip(unique_labels, counts)):
        print(f"  Class {label_idx}: {count} ({count/len(y)*100:.1f}%)")
    
    # Convert to class names for easier handling
    class_names = ['CentralApnea', 'Normal', 'ObstructiveApnea']  # Assuming 3-class after MixedApnea removal
    
    X_balanced = []
    y_balanced = []
    
    # Target counts for better balance (same logic as your original)
    target_counts = {
        0: 1000,  # CentralApnea
        1: 5000,  # Normal  
        2: 6000   # ObstructiveApnea
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
    """Extract advanced statistical features (same as your original)"""
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

def compute_feature_importance(X_train, y_train, channels):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_importance = sorted(zip(channels, importances), key=lambda x: x[1], reverse=True)

    return feature_importance

def select_top_features(feature_importance, percentage):
    top_n = int(len(feature_importance) * percentage)
    return feature_importance[:top_n]

def get_feature_importance_and_select(X_train, y_train, channels):
    feature_importance = compute_feature_importance(X_train, y_train, channels)
    
    top_25_percent = select_top_features(feature_importance, 0.25)
    top_50_percent = select_top_features(feature_importance, 0.50)

    return top_25_percent, top_50_percent

def get_feature_importance(X, y, channels):
    """Get feature importance using Random Forest with same config as original"""
    print("🔍 EXTRACTING ADVANCED FEATURES FOR RANDOM FOREST")
    print("=" * 60)
    
    # Extract advanced features for each channel (same as your original)
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
        
        if (sample_idx + 1) % 1000 == 0:
            print(f"  Processed {sample_idx + 1}/{X.shape[0]} samples")
    
    X_features = np.array(X_features)
    print(f"Feature extraction complete: {X_features.shape}")
    print(f"Total features: {len(feature_names)}")
    
    # Balance classes (same as your original)
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
    
    # Train Random Forest with same config as original
    print("\n🌲 TRAINING RANDOM FOREST")
    print("-" * 40)
    
    rf_config = CONFIG['rf_config'].copy()
    rf_config['class_weight'] = class_weight_dict
    
    rf = RandomForestClassifier(**rf_config)
    rf.fit(X_balanced, y_rf)
    
    # Get feature importance
    importances = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\n📊 TOP 20 FEATURE IMPORTANCES:")
    print("-" * 50)
    for i, (feature, importance) in enumerate(feature_importance[:20], 1):
        print(f"{i:2d}. {feature:30s}: {importance:.4f}")
    
    # Channel-level importance (same as your original)
    channel_importance = {}
    for feature, importance in feature_importance:
        channel = feature.split('_')[0] + '_' + feature.split('_')[1]  # Handle "EEG A1-A2" format
        if channel not in channel_importance:
            channel_importance[channel] = 0
        channel_importance[channel] += importance
    
    print(f"\n🔬 CHANNEL-LEVEL IMPORTANCE:")
    print("-" * 40)
    for channel, importance in sorted(channel_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{channel:15s}: {importance:.4f}")
    
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
        # Extract channel name from feature name (e.g., "EEG A1-A2_mean" -> "EEG A1-A2")
        parts = feature_name.split('_')
        if len(parts) >= 2:
            channel_name = '_'.join(parts[:-1])  # Everything except the last part (which is the feature type)
            
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