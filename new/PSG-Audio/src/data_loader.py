# data_loader.py

import os
import pickle
import numpy as np
import gc
import psutil
from config.config import CONFIG

def monitor_memory_usage():
    """Monitor memory usage if enabled"""
    if CONFIG['monitoring']['log_memory_usage']:
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.used/1e9:.1f}GB used / {memory.total/1e9:.1f}GB total ({memory.percent:.1f}%)")

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

def get_cache_path(data_path, channels, max_segments):
    """Generate cache file path based on data configuration"""
    # Use existing preprocessed cache file directly
    existing_cache = "/mnt/c/Users/Ayman/Downloads/cache/285_patients_processed_ch-8238238516549525776_full_cached.npz"
    
    # Check if the existing cache exists
    if os.path.exists(existing_cache):
        return existing_cache
    
    # Fallback: Create cache directory in the src folder (local to project)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(current_dir, 'data_cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create unique cache name based on parameters
    data_filename = os.path.basename(data_path).replace('.pkl', '')
    channels_hash = abs(hash(tuple(sorted(channels))))  # Use abs() for consistent naming
    max_seg_str = f"_{max_segments}" if max_segments else "_full"
    cache_filename = f"{data_filename}_ch{channels_hash}{max_seg_str}_cached.npz"
    
    return os.path.join(cache_dir, cache_filename)

def save_cached_data(cache_path, X, y):
    """Save processed data to cache"""
    print(f"💾 Saving processed data to cache: {cache_path}")
    try:
        np.savez_compressed(cache_path, X=X, y=y)
        cache_size = os.path.getsize(cache_path) / (1024**3)
        print(f"✅ Cache saved successfully ({cache_size:.2f} GB)")
        return True
    except Exception as e:
        print(f"⚠️  Failed to save cache: {e}")
        return False

def load_cached_data(cache_path):
    """Load processed data from cache"""
    print(f"📂 Loading cached data from: {cache_path}")
    try:
        cache_size = os.path.getsize(cache_path) / (1024**3)
        print(f"   Cache file size: {cache_size:.2f} GB")
        
        data = np.load(cache_path)
        X = data['X']
        y = data['y']
        
        print(f"✅ Cache loaded successfully!")
        print(f"   Shape: {X.shape}")
        print(f"   Memory: {X.nbytes / 1e9:.1f} GB")
        
        return X, y
    except Exception as e:
        print(f"⚠️  Failed to load cache: {e}")
        return None, None

def load_data_streaming(data_path, channels, max_segments=None):
    """Optimized data loader for 1TB RAM system with caching support"""
    print(f"🚀 OPTIMIZED DATA LOADING (1TB RAM System)")
    print(f"Data path: {data_path}")
    print(f"Target channels: {channels}")
    print(f"Max segments: {max_segments}")
    print(f"Load full dataset in memory: {CONFIG['data_loading']['load_full_in_memory']}")
    print(f"Cache enabled: {CONFIG['data_loading'].get('cache_dataset', True)}")
    
    monitor_memory_usage()
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Check for cached data
    cache_enabled = CONFIG['data_loading'].get('cache_dataset', True)
    if cache_enabled:
        cache_path = get_cache_path(data_path, channels, max_segments)
        print(f"\n🔍 Checking for cached data...")
        
        if os.path.exists(cache_path):
            # Check if cache is newer than source file
            cache_mtime = os.path.getmtime(cache_path)
            source_mtime = os.path.getmtime(data_path)
            
            if cache_mtime > source_mtime:
                print("✅ Found valid cache file!")
                X, y = load_cached_data(cache_path)
                if X is not None and y is not None:
                    monitor_memory_usage()
                    return X, y
                else:
                    print("⚠️  Cache loading failed, will reload from source")
            else:
                print("⚠️  Cache is older than source file, will reload from source")
        else:
            print("📝 No cache found, will create one after loading")
    else:
        print("📝 Cache disabled in config")
    
    # Check file size
    file_size = os.path.getsize(data_path) / (1024**3)  # Convert to GB
    print(f"File size: {file_size:.2f} GB")
    
    # Load segments - optimized for large RAM with error handling
    print("Loading data file...")
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    except EOFError as e:
        raise ValueError(f"Pickle file is corrupted or incomplete. The file may have been truncated during creation or transfer. Error: {e}")
    except Exception as e:
        raise ValueError(f"Error loading pickle file: {e}. The file may be corrupted.")
    
    if isinstance(data, list):
        segments = data
        total_segments = len(segments)
    else:
        raise ValueError("Expected list format")
    
    if max_segments:
        segments = segments[:max_segments]
        total_segments = len(segments)
    
    print(f"Processing {total_segments} segments from 285-patient dataset...")
    monitor_memory_usage()
    
    # With 1TB RAM, we can process much larger batches
    batch_size = 5000 if CONFIG['data_loading']['load_full_in_memory'] else 500
    print(f"Using optimized batch size: {batch_size}")
    
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
        
        # Memory management - less aggressive with 1TB RAM
        if not CONFIG['data_loading']['load_full_in_memory']:
            del batch, X_batch
            gc.collect()
        
        if (i // batch_size + 1) % 5 == 0:  # More frequent updates for large batches
            print(f"  Processed {min(i+batch_size, total_segments)}/{total_segments} segments")
            monitor_memory_usage()
    
    # Clear original segments only if not caching
    if not CONFIG['data_loading']['cache_dataset']:
        del segments, data
        gc.collect()
    
    if not X_batches:
        raise ValueError("No valid segments found!")
    
    # Combine batches
    print("Combining batches with optimized memory usage...")
    X = np.concatenate(X_batches, axis=0)
    y = np.array(y_batches)
    
    print(f"Final data loaded from 285-patient dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Memory: {X.nbytes / 1e9:.1f} GB")
    print(f"  Valid segments: {valid_count}")
    print(f"  Invalid segments: {invalid_count}")
    
    monitor_memory_usage()
    
    # Show class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(y)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Save to cache if enabled
    if cache_enabled:
        print(f"\n💾 Creating cache for future runs...")
        save_cached_data(cache_path, X, y)
    
    return X, y

def preprocess_data(X):
    """Optimized standardization for large datasets"""
    print("Standardizing data with optimized processing...")
    X_std = np.copy(X)
    
    # Parallel processing for large datasets
    for i in range(X.shape[-1]):
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
        print(f"  Channel {i}: standardized (mean={mean:.3f}, std={std:.3f})")
    
    return X_std

def validate_data_quality(X, y):
    """Enhanced data quality validation with memory monitoring"""
    print("\n🔍 ENHANCED DATA QUALITY CHECK (285-patient dataset):")
    print("-" * 60)
    
    monitor_memory_usage()
    
    # Check for data imbalance
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Check for extreme imbalance
    imbalance_ratio = max(counts) / min(counts)
    if imbalance_ratio > 10:
        print(f"⚠️  SEVERE CLASS IMBALANCE: {imbalance_ratio:.1f}:1")
        print("   Using balanced class weights and advanced sampling")
    
    # Enhanced data validation
    print(f"Data shape: {X.shape}")
    print(f"Data memory: {X.nbytes / 1e9:.2f} GB")
    print(f"Data range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Data std: {X.std():.3f}")
    
    # Check for NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Data issues: {nan_count} NaNs, {inf_count} Infs")
    else:
        print("✅ No NaN or Inf values found")
    
    # Memory efficiency check
    expected_memory = X.shape[0] * X.shape[1] * X.shape[2] * 4 / 1e9  # 4 bytes per float32
    actual_memory = X.nbytes / 1e9
    efficiency = actual_memory / expected_memory
    print(f"Memory efficiency: {efficiency:.2f} (1.0 = optimal)")
    
    return imbalance_ratio