import os
import sys
import pickle
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from random_forest import (
    get_feature_importance,
    select_top_features,
    convert_feature_names_to_channels,
    monitor_memory_usage
)
from config.config import CONFIG

def get_channels_from_pickle(data_path):
    """Extract channel names from pickle file"""
    print("Auto-detecting channels from pickle file...")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Expected non-empty list of segments")
    
    # Get first valid segment
    first_segment = None
    for seg in data:
        if isinstance(seg, dict) and 'features' in seg and 'label' in seg:
            first_segment = seg
            break
    
    if first_segment is None:
        raise ValueError("No valid segments found with 'features' and 'label' keys")
    
    # Extract channel names from features dictionary
    channels = list(first_segment['features'].keys())
    
    print(f"Auto-detected {len(channels)} channels: {channels}")
    
    return channels

def load_pickle_data(data_path, channels=None, max_segments=None, exclude_classes=None):
    """Load and process pickle file data"""
    print(f"\n{'='*70}")
    print(f"LOADING DATA FROM PICKLE FILE")
    print(f"{'='*70}")
    print(f"Data path: {data_path}")
    print(f"Max segments: {max_segments if max_segments else 'All'}")
    print(f"Exclude classes: {exclude_classes if exclude_classes else 'None'}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load pickle file
    print("\nLoading pickle file...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Expected list of segments in pickle file")
    
    print(f"Loaded {len(data):,} segments from pickle file")
    
    # Auto-detect channels if not provided
    if channels is None:
        channels = get_channels_from_pickle(data_path)
    
    print(f"Using {len(channels)} channels: {channels}")
    
    # Process segments
    X_list = []
    y_list = []
    valid_count = 0
    invalid_count = 0
    excluded_count = 0
    label_counts = {}
    
    print("\nProcessing segments...")
    monitor_memory_usage()
    
    for i, seg in enumerate(data):
        if not isinstance(seg, dict):
            invalid_count += 1
            continue
        
        # Check for required keys
        if 'features' not in seg or 'label' not in seg:
            invalid_count += 1
            continue
        
        label = seg['label']
        
        # Check if segment should be excluded
        if exclude_classes and label in exclude_classes:
            excluded_count += 1
            continue
        
        # Count labels
        label_counts[label] = label_counts.get(label, 0) + 1
        
        # Extract channel data from features dictionary
        channel_data = []
        valid_segment = True
        
        for ch in channels:
            if ch in seg['features'] and seg['features'][ch] is not None:
                ch_data = np.array(seg['features'][ch], dtype=np.float32)
                
                # Handle NaN values
                if np.any(np.isnan(ch_data)):
                    ch_data = np.nan_to_num(ch_data, nan=0.0)
                
                if len(ch_data) > 0:
                    channel_data.append(ch_data)
                else:
                    valid_segment = False
                    break
            else:
                valid_segment = False
                break
        
        if valid_segment and len(channel_data) == len(channels):
            # Ensure all channels have same length
            min_len = min(len(ch) for ch in channel_data)
            if min_len > 100:  # Minimum viable segment length
                # Truncate to same length and transpose
                channel_data = [ch[:min_len] for ch in channel_data]
                X_list.append(np.array(channel_data).T)  # Shape: (time_points, channels)
                y_list.append(label)
                valid_count += 1
            else:
                invalid_count += 1
        else:
            invalid_count += 1
        
        # Progress update
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,}/{len(data):,} - Valid: {valid_count:,}, Invalid: {invalid_count:,}, Excluded: {excluded_count:,}")
            monitor_memory_usage()
        
        # Limit segments if specified
        if max_segments and valid_count >= max_segments:
            print(f"\nReached max segments limit: {max_segments:,}")
            break
    
    print(f"\nFinal processing results:")
    print(f"  Valid: {valid_count:,}")
    print(f"  Invalid: {invalid_count:,}")
    print(f"  Excluded: {excluded_count:,}")
    
    if not X_list:
        raise ValueError("No valid segments found!")
    
    # Convert to numpy arrays
    print("\nConverting to numpy arrays...")
    X = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list)
    
    print(f"Data shape: {X.shape}")
    print(f"Memory usage: {X.nbytes / 1e6:.1f} MB")
    
    # Encode labels to categorical
    print("\nEncoding labels...")
    unique_labels = sorted(set(y_raw))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"Label mapping:")
    for label, idx in label_to_idx.items():
        count = label_counts.get(label, 0)
        print(f"  {idx}: {label} ({count:,} samples, {count/valid_count*100:.1f}%)")
    
    # Convert to categorical
    y_encoded = np.array([label_to_idx[label] for label in y_raw])
    y_categorical = np.eye(len(unique_labels))[y_encoded]
    
    print(f"\nFinal encoded data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y_categorical.shape}")
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Classes: {unique_labels}")
    
    monitor_memory_usage()
    
    return X, y_categorical, channels, unique_labels

def test_random_forest():
    """Test Random Forest feature importance extraction"""
    start_time = datetime.now()
    
    print("\n" + "="*70)
    print("RANDOM FOREST FEATURE IMPORTANCE TEST")
    print("="*70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    data_path = "/raid/userdata/cybulskm/ThesisProj/285_patients_processed_v2.pkl"
    max_segments = CONFIG.get('data_loading', {}).get('max_segments_per_patient', 20000)
    exclude_classes = ['MixedApnea']  # Exclude MixedApnea for 3-class problem
    feature_percentage = 0.25  # Top 25% features
    
    print(f"\nConfiguration:")
    print(f"  Data path: {data_path}")
    print(f"  Max segments: {max_segments if max_segments else 'All'}")
    print(f"  Exclude classes: {exclude_classes}")
    print(f"  Feature selection: Top {feature_percentage*100:.0f}%")
    
    try:
        # Load data
        X, y, channels, class_labels = load_pickle_data(
            data_path, 
            channels=None,  # Auto-detect
            max_segments=max_segments,
            exclude_classes=exclude_classes
        )
        
        print(f"\n{'='*70}")
        print("DATA LOADED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"  Samples: {X.shape[0]:,}")
        print(f"  Time points per sample: {X.shape[1]:,}")
        print(f"  Channels: {X.shape[2]} ({', '.join(channels)})")
        print(f"  Classes: {len(class_labels)} ({', '.join(class_labels)})")
        
        # Get feature importance using Random Forest
        print(f"\n{'='*70}")
        print("RUNNING RANDOM FOREST FEATURE IMPORTANCE")
        print(f"{'='*70}")
        
        feature_importance = get_feature_importance(X, y, channels)
        
        # Select top features
        print(f"\n{'='*70}")
        print("SELECTING TOP FEATURES")
        print(f"{'='*70}")
        
        top_features = select_top_features(feature_importance, percentage=feature_percentage)
        
        # Convert to channel names
        print(f"\n{'='*70}")
        print("CONVERTING FEATURES TO CHANNELS")
        print(f"{'='*70}")
        
        selected_channels = convert_feature_names_to_channels(top_features, channels)
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Total channels available: {len(channels)}")
        print(f"  Total features extracted: {len(feature_importance):,}")
        print(f"  Top features selected: {len(top_features):,} ({feature_percentage*100:.0f}%)")
        print(f"  Channels represented: {len(selected_channels)}/{len(channels)}")
        print(f"  Selected channels: {', '.join(selected_channels)}")
        
        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"rf_feature_importance_{timestamp}.txt")
        
        print(f"\nSaving results to: {results_file}")
        
        with open(results_file, 'w') as f:
            f.write("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Data path: {data_path}\n")
            f.write(f"Samples: {X.shape[0]:,}\n")
            f.write(f"Channels: {len(channels)} ({', '.join(channels)})\n")
            f.write(f"Classes: {len(class_labels)} ({', '.join(class_labels)})\n")
            f.write(f"Excluded classes: {exclude_classes}\n\n")
            
            f.write("TOP 50 FEATURE IMPORTANCES:\n")
            f.write("-"*60 + "\n")
            for i, (feature, importance) in enumerate(feature_importance[:50], 1):
                f.write(f"{i:2d}. {feature:40s}: {importance:.6f}\n")
            
            f.write(f"\n\nSELECTED TOP {feature_percentage*100:.0f}% FEATURES ({len(top_features):,} features):\n")
            f.write("-"*60 + "\n")
            for i, feature in enumerate(top_features, 1):
                importance = next(imp for feat, imp in feature_importance if feat == feature)
                f.write(f"{i:3d}. {feature:40s}: {importance:.6f}\n")
            
            f.write(f"\n\nSELECTED CHANNELS ({len(selected_channels)} channels):\n")
            f.write("-"*60 + "\n")
            for i, channel in enumerate(selected_channels, 1):
                f.write(f"{i:2d}. {channel}\n")
        
        print(f"✅ Results saved successfully")
        
        # Final timing
        total_time = datetime.now() - start_time
        print(f"\n{'='*70}")
        print(f"TEST COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Total runtime: {total_time}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return selected_channels, feature_importance
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    selected_channels, feature_importance = test_random_forest()
    
    if selected_channels:
        print(f"\n✅ Test completed successfully!")
        print(f"Selected {len(selected_channels)} channels for CNN training")
    else:
        print(f"\n❌ Test failed!")
        sys.exit(1)