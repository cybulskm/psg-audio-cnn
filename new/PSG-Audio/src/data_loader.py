# data_loader.py

import os
import pickle
import numpy as np
import gc
from config.config import CONFIG

def filter_classes(X, y, exclude_classes):
    """Filter out specified classes"""
    if not exclude_classes:
        return X, y
    
    print(f"Filtering out classes: {exclude_classes}")
    
    # Create mask for classes to keep
    mask = np.array([label not in exclude_classes for label in y])
    
    X_filtered = X[mask]
    y_filtered = y[mask]
    
    print(f"Original samples: {len(X)}")
    print(f"Filtered samples: {len(X_filtered)}")
    print(f"Removed: {len(X) - len(X_filtered)}")
    
    # Show new class distribution
    unique_labels, counts = np.unique(y_filtered, return_counts=True)
    print("New class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count}")
    
    return X_filtered, y_filtered

def load_data_streaming(data_path, channels, max_segments=None):
    """Memory-efficient streaming loader with improved error handling"""
    print(f"Loading data from: {data_path}")
    print(f"Target channels: {channels}")
    print(f"Max segments: {max_segments}")
    print(f"Using 285-patient dataset")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load segments
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
    
    print(f"Processing {total_segments} segments from 285-patient dataset...")
    
    # Process in small batches to control memory
    batch_size = 500
    X_batches = []
    y_batches = []
    valid_count = 0
    invalid_count = 0
    
    for i in range(0, total_segments, batch_size):
        batch = segments[i:i+batch_size]
        X_batch = []
        y_batch = []
        
        for seg in batch:
            if not isinstance(seg, dict):
                invalid_count += 1
                continue
                
            # Process segment
            channel_data = []
            valid = True
            
            for ch in channels:
                if ch in seg and seg[ch] is not None and len(seg[ch]) > 0:
                    data_array = np.array(seg[ch], dtype=np.float32)
                    if np.any(np.isnan(data_array)):
                        data_array = np.nan_to_num(data_array, nan=0.0)
                    channel_data.append(data_array)
                else:
                    valid = False
                    break
            
            if valid and len(channel_data) == len(channels):
                min_len = min(len(ch) for ch in channel_data)
                if min_len > 1000:  # Reasonable minimum
                    channel_data = [ch[:min_len] for ch in channel_data]
                    X_batch.append(np.array(channel_data, dtype=np.float32).T)
                    y_batch.append(seg.get('Label', 'Unknown'))
                    valid_count += 1
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        if X_batch:
            X_batches.append(np.array(X_batch, dtype=np.float32))
            y_batches.extend(y_batch)
        
        # Clear batch from memory
        del batch, X_batch
        gc.collect()
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i+batch_size, total_segments)}/{total_segments} segments")
    
    # Clear original segments
    del segments, data
    gc.collect()
    
    if not X_batches:
        raise ValueError("No valid segments found!")
    
    # Combine batches
    print("Combining batches...")
    X = np.concatenate(X_batches, axis=0)
    y = np.array(y_batches)
    
    print(f"Final data loaded from 285-patient dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Memory: {X.nbytes / 1e6:.1f} MB")
    print(f"  Valid segments: {valid_count}")
    print(f"  Invalid segments: {invalid_count}")
    
    # Show class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(y)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return X, y

def preprocess_data(X):
    """Standardize data per channel"""
    X_std = np.copy(X)
    for i in range(X.shape[-1]):
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
    return X_std

def validate_data_quality(X, y):
    """Enhanced data quality validation"""
    print("\n🔍 DATA QUALITY CHECK (285-patient dataset):")
    print("-" * 50)
    
    # Check for data imbalance
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Check for extreme imbalance
    imbalance_ratio = max(counts) / min(counts)
    if imbalance_ratio > 10:
        print(f"⚠️  SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}:1")
        print("   Using balanced class weights in models")
    
    # Check data ranges
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Data std: {X.std():.3f}")
    
    # Check for NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Data issues: {nan_count} NaNs, {inf_count} Infs")
    else:
        print("✅ No NaN or Inf values found")
    
    return imbalance_ratio