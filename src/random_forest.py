import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import psutil
import time
from config.config import CONFIG

def monitor_memory_usage():
    """Monitor memory usage if enabled"""
    if CONFIG['monitoring']['log_memory_usage']:
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.used/1e9:.1f}GB used / {memory.total/1e9:.1f}GB total ({memory.percent:.1f}%)")

def extract_advanced_features(channel_data, fs=200):
    """
    Extract comprehensive features optimized for apnea detection
    
    Key improvements:
    - Frequency domain features (apneas have distinct spectral signatures)
    - Temporal dynamics (breathing patterns change)
    - Wavelet features (multi-resolution analysis)
    - Signal quality metrics
    """
    features = {}
    
    # ============================================
    # TIME DOMAIN FEATURES
    # ============================================
    
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
    features['p10'] = np.percentile(channel_data, 10)
    features['p25'] = np.percentile(channel_data, 25)
    features['p75'] = np.percentile(channel_data, 75)
    features['p90'] = np.percentile(channel_data, 90)
    features['iqr'] = features['p75'] - features['p25']
    
    # Higher-order statistics
    if features['std'] > 1e-10:
        normalized = (channel_data - features['mean']) / features['std']
        features['skew'] = stats.skew(normalized)
        features['kurtosis'] = stats.kurtosis(normalized)
    else:
        features['skew'] = 0
        features['kurtosis'] = 0
    
    # Signal characteristics
    features['zcr'] = np.sum(np.diff(np.sign(channel_data)) != 0) / len(channel_data)
    features['energy'] = np.sum(channel_data**2)
    features['power'] = features['energy'] / len(channel_data)
    
    # ============================================
    # TEMPORAL DYNAMICS (CRITICAL FOR APNEA)
    # ============================================
    
    # First and second derivatives (rate of change)
    diff1 = np.diff(channel_data)
    diff2 = np.diff(diff1)
    
    features['diff1_mean'] = np.mean(np.abs(diff1))
    features['diff1_std'] = np.std(diff1)
    features['diff2_mean'] = np.mean(np.abs(diff2))
    features['diff2_std'] = np.std(diff2)
    
    # Slope changes (breathing pattern transitions)
    slope_changes = np.sum(np.diff(np.sign(diff1)) != 0) / len(diff1)
    features['slope_changes'] = slope_changes
    
    # Peak detection (breathing cycles)
    try:
        peaks, _ = signal.find_peaks(channel_data, distance=fs//2)  # Min 0.5s between peaks
        features['num_peaks'] = len(peaks)
        features['peak_rate'] = len(peaks) / (len(channel_data) / fs)  # peaks per second
        
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            features['peak_interval_mean'] = np.mean(peak_intervals)
            features['peak_interval_std'] = np.std(peak_intervals)
            features['peak_interval_cv'] = features['peak_interval_std'] / features['peak_interval_mean'] if features['peak_interval_mean'] > 0 else 0
        else:
            features['peak_interval_mean'] = 0
            features['peak_interval_std'] = 0
            features['peak_interval_cv'] = 0
    except:
        features['num_peaks'] = 0
        features['peak_rate'] = 0
        features['peak_interval_mean'] = 0
        features['peak_interval_std'] = 0
        features['peak_interval_cv'] = 0
    
    # ============================================
    # FREQUENCY DOMAIN FEATURES (SPECTRAL)
    # ============================================
    
    # FFT for spectral analysis
    n = len(channel_data)
    yf = fft(channel_data)
    xf = fftfreq(n, 1/fs)
    
    # Only positive frequencies
    pos_mask = xf > 0
    freqs = xf[pos_mask]
    power_spectrum = np.abs(yf[pos_mask])**2
    
    # Frequency bands (relevant for sleep apnea)
    # Delta: 0.5-4 Hz, Theta: 4-8 Hz, Alpha: 8-13 Hz, Beta: 13-30 Hz
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'low': (0.1, 1),      # Very low frequency (respiratory)
        'breathing': (0.1, 0.5)  # Normal breathing rate
    }
    
    total_power = np.sum(power_spectrum)
    
    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(power_spectrum[band_mask])
        features[f'power_{band_name}'] = band_power
        features[f'power_{band_name}_rel'] = band_power / total_power if total_power > 0 else 0
    
    # Spectral features
    if total_power > 0:
        features['spectral_centroid'] = np.sum(freqs * power_spectrum) / total_power
        features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * power_spectrum) / total_power)
        features['spectral_rolloff'] = freqs[np.where(np.cumsum(power_spectrum) >= 0.85 * total_power)[0][0]] if len(np.where(np.cumsum(power_spectrum) >= 0.85 * total_power)[0]) > 0 else 0
        features['spectral_flatness'] = stats.gmean(power_spectrum + 1e-10) / (np.mean(power_spectrum) + 1e-10)
    else:
        features['spectral_centroid'] = 0
        features['spectral_spread'] = 0
        features['spectral_rolloff'] = 0
        features['spectral_flatness'] = 0
    
    # Dominant frequency
    if len(power_spectrum) > 0:
        features['dominant_freq'] = freqs[np.argmax(power_spectrum)]
    else:
        features['dominant_freq'] = 0
    
    # ============================================
    # ENTROPY MEASURES (SIGNAL COMPLEXITY)
    # ============================================
    
    # Sample entropy (regularity of signal)
    try:
        features['sample_entropy'] = _sample_entropy(channel_data, m=2, r=0.2*features['std'])
    except:
        features['sample_entropy'] = 0
    
    # Approximate entropy
    try:
        features['approx_entropy'] = _approximate_entropy(channel_data, m=2, r=0.2*features['std'])
    except:
        features['approx_entropy'] = 0
    
    # Spectral entropy
    psd_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) > 0 else power_spectrum
    psd_norm = psd_norm[psd_norm > 0]
    features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
    
    # ============================================
    # SIGNAL QUALITY INDICATORS
    # ============================================
    
    # Saturation detection (clipping)
    features['saturation_rate'] = np.sum((channel_data >= 0.95 * features['max']) | 
                                         (channel_data <= 0.95 * features['min'])) / len(channel_data)
    
    # Signal-to-noise ratio estimate
    if features['std'] > 0:
        features['snr_estimate'] = features['mean'] / features['std']
    else:
        features['snr_estimate'] = 0
    
    # Autocorrelation at lag 1 (signal smoothness)
    if len(channel_data) > 1:
        features['autocorr_lag1'] = np.corrcoef(channel_data[:-1], channel_data[1:])[0, 1]
    else:
        features['autocorr_lag1'] = 0
    
    return features

def _sample_entropy(data, m, r):
    """Calculate sample entropy"""
    N = len(data)
    
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1 for x_i in x]
        return sum(C)
    
    return -np.log(_phi(m + 1) / _phi(m)) if _phi(m) > 0 and _phi(m+1) > 0 else 0

def _approximate_entropy(data, m, r):
    """Calculate approximate entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    
    def _phi(m):
        N = len(data)
        x = [[data[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    return abs(_phi(m) - _phi(m + 1))

def balance_classes_smarter(X, y, strategy='moderate'):
    """
    Smarter balancing that doesn't over-duplicate
    
    Key changes:
    - Less aggressive upsampling
    - Preserve more minority class diversity
    - Don't add noise (causes issues with RF)
    """
    print(f"🔄 SMART CLASS BALANCING")
    print("-" * 50)
    
    y_labels = np.argmax(y, axis=1)
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    
    print("Original class distribution:")
    for label_idx, count in zip(unique_labels, counts):
        print(f"  Class {label_idx}: {count:,} ({count/len(y)*100:.1f}%)")
    
    # Calculate target based on median count (more conservative)
    median_count = int(np.median(counts))
    max_count = int(np.max(counts))
    
    # Target: bring minority up to 70% of median, cap majority at 150% of median
    target_min = int(median_count * 0.7)
    target_max = int(median_count * 1.5)
    
    print(f"\nTarget range: {target_min:,} to {target_max:,} samples per class")
    
    X_balanced = []
    y_balanced = []
    
    for label_idx in unique_labels:
        mask = (y_labels == label_idx)
        X_class = X[mask]
        y_class = y[mask]
        current_count = len(X_class)
        
        if current_count < target_min:
            # Moderate upsampling
            target_count = min(target_min, current_count * 3)  # Don't more than triple
            indices = np.random.choice(len(X_class), target_count, replace=True)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            print(f"  Class {label_idx}: {current_count:,} -> {target_count:,} (upsampled moderately)")
        elif current_count > target_max:
            # Downsample majority
            indices = np.random.choice(len(X_class), target_max, replace=False)
            X_resampled = X_class[indices]
            y_resampled = y_class[indices]
            print(f"  Class {label_idx}: {current_count:,} -> {target_max:,} (downsampled)")
        else:
            X_resampled, y_resampled = X_class, y_class
            print(f"  Class {label_idx}: {current_count:,} (kept)")
        
        X_balanced.append(X_resampled)
        y_balanced.append(y_resampled)
    
    X_final = np.vstack(X_balanced)
    y_final = np.vstack(y_balanced)
    
    # Shuffle
    indices = np.random.permutation(len(X_final))
    X_final = X_final[indices]
    y_final = y_final[indices]
    
    print(f"\nFinal balanced data shape: {X_final.shape}")
    
    # Final distribution
    y_final_labels = np.argmax(y_final, axis=1)
    unique_final, counts_final = np.unique(y_final_labels, return_counts=True)
    print("Final distribution:")
    for label_idx, count in zip(unique_final, counts_final):
        print(f"  Class {label_idx}: {count:,} ({count/len(y_final)*100:.1f}%)")
    
    return X_final, y_final

def get_feature_importance(X, y, channels):
    """Get feature importance using optimized Random Forest"""
    print("🌲 ENHANCED RANDOM FOREST FEATURE EXTRACTION")
    print("=" * 70)
    
    start_time = time.time()
    
    # Extract advanced features
    feature_names = []
    X_features = []
    
    print("Extracting enhanced apnea-specific features...")
    monitor_memory_usage()
    
    batch_size = 5000
    
    for batch_start in range(0, X.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X.shape[0])
        batch_features = []
        
        for sample_idx in range(batch_start, batch_end):
            sample_features = []
            
            for ch_idx, channel_name in enumerate(channels):
                channel_data = X[sample_idx, :, ch_idx]
                features = extract_advanced_features(channel_data, fs=200)
                
                if sample_idx == batch_start and ch_idx == 0:
                    # Log feature count for first sample
                    print(f"  Extracting {len(features)} features per channel")
                
                if sample_idx == 0:
                    for feat_name in features.keys():
                        feature_names.append(f"{channel_name}_{feat_name}")
                
                sample_features.extend(features.values())
            
            batch_features.append(sample_features)
        
        X_features.extend(batch_features)
        
        if batch_end % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processed {batch_end:,}/{X.shape[0]:,} ({elapsed:.1f}s)")
            monitor_memory_usage()
    
    X_features = np.array(X_features, dtype=np.float32)
    print(f"Feature extraction complete: {X_features.shape}")
    print(f"Total features: {len(feature_names)}")
    
    # Smarter balancing
    X_balanced, y_balanced = balance_classes_smarter(X_features, y)
    
    y_rf = np.argmax(y_balanced, axis=1)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_rf), y=y_rf)
    class_weight_dict = dict(zip(np.unique(y_rf), class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Enhanced Random Forest configuration
    print("\n🚀 TRAINING ENHANCED RANDOM FOREST")
    print("-" * 50)
    
    rf_config = {
        'n_estimators': 300,  # More trees for better estimates
        'max_depth': 20,      # Deeper trees
        'min_samples_split': 10,  # Less aggressive pruning
        'min_samples_leaf': 4,
        'max_features': 'sqrt',  # Standard for classification
        'class_weight': class_weight_dict,
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 1
    }
    
    print("Configuration:")
    for key, value in rf_config.items():
        print(f"  {key}: {value}")
    
    rf_start = time.time()
    rf = RandomForestClassifier(**rf_config)
    rf.fit(X_balanced, y_rf)
    rf_time = time.time() - rf_start
    
    print(f"\n✅ Training completed in {rf_time:.1f}s")
    print(f"   OOB Score: {rf.oob_score_:.4f}")
    
    # Cross-validation with stratification
    print(f"\n🔄 STRATIFIED CROSS-VALIDATION (5 folds)")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_balanced, y_rf, cv=skf, n_jobs=-1, verbose=1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # Feature importance
    importances = rf.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    # Analysis
    print(f"\n📊 FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 50)
    print(f"Total features: {len(feature_importance):,}")
    print(f"Top feature importance: {feature_importance[0][1]:.6f}")
    print(f"Median importance: {np.median([imp for _, imp in feature_importance]):.6f}")
    
    print("\n📋 TOP 30 FEATURES:")
    print("-" * 60)
    for i, (feature, importance) in enumerate(feature_importance[:30], 1):
        print(f"{i:2d}. {feature:45s}: {importance:.6f}")
    
    # Channel-level importance
    channel_importance = {}
    for feature_name, importance in feature_importance:
        parts = feature_name.rsplit('_', 1)  # Split from right
        if len(parts) == 2:
            channel_name = parts[0]
            if channel_name in channels:
                channel_importance[channel_name] = channel_importance.get(channel_name, 0) + importance
    
    print(f"\n🎯 CHANNEL-LEVEL IMPORTANCE:")
    print("-" * 50)
    sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (channel, importance) in enumerate(sorted_channels, 1):
        print(f"{i:2d}. {channel:20s}: {importance:.6f}")
    
    monitor_memory_usage()
    print(f"\nTotal time: {time.time() - start_time:.1f}s")
    
    return feature_importance

def select_top_features(feature_importance, percentage=0.25):
    """Select top features by percentage"""
    n_features = max(1, int(len(feature_importance) * percentage))
    top_features = [feat for feat, _ in feature_importance[:n_features]]
    
    print(f"\n🎯 SELECTED TOP {percentage*100:.0f}% FEATURES ({len(top_features):,} features)")
    
    return top_features

def convert_feature_names_to_channels(feature_names, channels):
    """Convert feature names to unique channel list"""
    channel_set = set()
    
    for feature_name in feature_names:
        parts = feature_name.rsplit('_', 1)
        if len(parts) == 2:
            channel_name = parts[0]
            if channel_name in channels:
                channel_set.add(channel_name)
    
    selected_channels = [ch for ch in channels if ch in channel_set]
    
    print(f"\n📍 MAPPED TO {len(selected_channels)} CHANNELS:")
    for i, channel in enumerate(selected_channels, 1):
        print(f"{i:2d}. {channel}")
    
    return selected_channels