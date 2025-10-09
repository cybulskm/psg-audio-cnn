import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
from datetime import datetime

# Add parent directory to path for config imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Ensure TensorFlow can use GPUs and configure memory growth
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Optionally enable mixed precision if available
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled: mixed_float16")
    except Exception:
        pass
    print(f"TensorFlow {tf.__version__} - GPUs detected: {len(gpus)}")
except Exception as e:
    print("TensorFlow GPU setup warning:", e)

# Simple import fix - try both import methods
try:
    # When running with python -m src.main
    from src.data_loader import load_data_streaming, filter_classes, validate_data_quality
    from src.random_forest import get_feature_importance, select_top_features, convert_feature_names_to_channels
    from src.cnn import train_and_evaluate_cnn
except ImportError:
    # When running from src directory: python main.py
    from data_loader import load_data_streaming, filter_classes, validate_data_quality
    from random_forest import get_feature_importance, select_top_features, convert_feature_names_to_channels
    from cnn import train_and_evaluate_cnn

from config.config import CONFIG

def setup_optimized_environment():
    """Setup optimized environment"""
    print("=" * 80)
    print("🚀 PSG-AUDIO CHANNEL ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Dataset: 285-patient dataset ({CONFIG['data_path']})")
    print("=" * 80)
    
    import tensorflow as tf
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # Detailed GPU check
    physical_gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    visible_devices = tf.config.get_visible_devices()
    
    print(f"\n🔍 GPU STATUS:")
    print(f"  Physical GPUs: {len(physical_gpus)}")
    print(f"  Logical GPUs: {len(logical_gpus)}")
    print(f"  Visible devices: {[d.device_type for d in visible_devices]}")
    
    if physical_gpus:
        for i, gpu in enumerate(physical_gpus):
            print(f"  GPU {i}: {gpu.name}")
        print(f"✅ GPU IS AVAILABLE AND WILL BE USED FOR TRAINING")
    else:
        print(f"❌ NO GPU DETECTED - WILL USE CPU ONLY")
    
    # Test GPU with a simple operation
    if physical_gpus:
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            print(f"✅ GPU test operation successful: GPU is functioning properly")
        except Exception as e:
            print(f"⚠️ GPU test failed: {e}")
    
    print("=" * 80)

def get_channel_importance_ranking(feature_importance, channels):
    """Get channel-level importance ranking"""
    channel_importance = {}
    
    # Sum up feature importances by channel
    for feature_name, importance in feature_importance:
        parts = feature_name.split('_')
        if len(parts) >= 2:
            channel_name = '_'.join(parts[:-1])
            if channel_name in channels:
                if channel_name not in channel_importance:
                    channel_importance[channel_name] = 0
                channel_importance[channel_name] += importance
    
    # Sort channels by importance
    sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 CHANNEL IMPORTANCE RANKING:")
    print("-" * 50)
    for i, (channel, importance) in enumerate(sorted_channels, 1):
        print(f"{i:2d}. {channel:<15s}: {importance:.6f}")
    
    return sorted_channels

def create_channel_groups(sorted_channels):
    """Create channel groups based on importance"""
    total_channels = len(sorted_channels)
    
    # Define group sizes based on 9 channels
    groups = {
        'tier_1_top3': sorted_channels[:3],      # Top 3 channels (33%)
        'tier_2_mid3': sorted_channels[3:6],     # Middle 3 channels (33%)  
        'tier_3_bottom3': sorted_channels[6:9],  # Bottom 3 channels (33%)
        'top_5_channels': sorted_channels[:5],   # Top 5 channels (56%)
        'top_7_channels': sorted_channels[:7],   # Top 7 channels (78%)
        'all_9_channels': sorted_channels[:9]    # All 9 channels (100%)
    }
    
    print(f"\n📊 CHANNEL GROUPING STRATEGY:")
    print("-" * 50)
    
    for group_name, channel_list in groups.items():
        channels_only = [ch for ch, _ in channel_list]
        importance_sum = sum(imp for _, imp in channel_list)
        print(f"{group_name:15s}: {len(channels_only)} channels, importance={importance_sum:.4f}")
        print(f"                 {channels_only}")
    
    return groups

def main():
    """Main function with channel-focused analysis"""
    pipeline_start = time.time()
    setup_optimized_environment()
    
    print("🧬 PSG-AUDIO: CHANNEL-BASED ANALYSIS PIPELINE")
    print("🔬 285-PATIENT DATASET")
    print("=" * 80)

    # STEP 1: Load Data
    print("\n📂 STEP 1: DATA LOADING")
    print("-" * 50)
    
    step_start = time.time()
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], CONFIG['max_segments'])
    
    labels, y_encoded = np.unique(y, return_inverse=True)
    print(f"✅ Data loaded: {X.shape}, Labels: {labels}")
    
    imbalance_ratio = validate_data_quality(X, y_encoded)
    step_time = time.time() - step_start
    print(f"   Loading time: {step_time:.1f}s")
    
    y_cat = to_categorical(y_encoded)

    # STEP 2: Random Forest Analysis
    print("\n🌲 STEP 2: RANDOM FOREST CHANNEL ANALYSIS")
    print("-" * 50)
    
    step_start = time.time()
    feature_importance = get_feature_importance(X, y_cat, CONFIG['channels'])
    step_time = time.time() - step_start
    print(f"✅ Random Forest completed in {step_time:.1f}s")

    # STEP 3: Channel-Based Grouping
    print("\n📊 STEP 3: CHANNEL IMPORTANCE & GROUPING")
    print("-" * 50)
    
    # Get channel importance ranking
    sorted_channels = get_channel_importance_ranking(feature_importance, CONFIG['channels'])
    
    # Create channel groups
    channel_groups = create_channel_groups(sorted_channels)

    # STEP 4: Prepare Channel-Based Datasets
    print("\n🔧 STEP 4: CHANNEL-BASED DATASET PREPARATION")
    print("-" * 50)
    
    datasets = {}
    
    for group_name, channel_list in channel_groups.items():
        channels_only = [ch for ch, _ in channel_list]
        X_subset, y_subset = prepare_channel_data_for_cnn(X, y_cat, channels_only, CONFIG['channels'])
        
        datasets[group_name] = {
            'X': X_subset,
            'y': y_subset,
            'channels': channels_only,
            'n_channels': len(channels_only),
            'total_importance': sum(imp for _, imp in channel_list)
        }
        print(f"   {group_name:15s}: {X_subset.shape}")

    # STEP 5: CNN Training on Channel Groups
    print("\n🤖 STEP 5: CNN TRAINING ON CHANNEL GROUPS")
    print("-" * 50)
    
    results = {}
    training_order = ['tier_1_top3', 'top_5_channels', 'top_7_channels', 'all_9_channels', 'tier_2_mid3', 'tier_3_bottom3']
    
    for i, group_name in enumerate(training_order, 1):
        dataset = datasets[group_name]
        
        print(f"\n{'='*15} CNN {i}/6: {group_name.upper()} {'='*15}")
        print(f"Channels ({dataset['n_channels']}): {dataset['channels']}")
        print(f"Combined importance: {dataset['total_importance']:.4f}")
        
        step_start = time.time()
        
        test_acc, model, history = train_and_evaluate_cnn(
            dataset['X'], dataset['y'],
            f"{group_name} ({dataset['n_channels']} channels)"
        )
        
        step_time = time.time() - step_start
        
        results[group_name] = {
            'accuracy': test_acc,
            'n_channels': dataset['n_channels'],
            'channels': dataset['channels'],
            'training_time': step_time,
            'total_importance': dataset['total_importance']
        }
        
        print(f"✅ {group_name} completed: {test_acc:.4f} accuracy ({step_time:.1f}s)")

    # STEP 6: Channel Group Analysis
    print("\n" + "="*80)
    print("🎯 CHANNEL GROUP PERFORMANCE ANALYSIS")
    print("="*80)
    
    total_time = time.time() - pipeline_start
    
    # Performance table
    print(f"\n📊 CHANNEL GROUP COMPARISON:")
    print("-" * 90)
    print(f"{'Group':<15} {'Channels':<10} {'Accuracy':<12} {'Importance':<12} {'Time':<10} {'Efficiency':<12}")
    print("-" * 90)
    
    baseline_acc = results['all_9_channels']['accuracy']
    
    for group_name in training_order:
        result = results[group_name]
        efficiency = result['accuracy'] / result['n_channels']  # Accuracy per channel
        
        print(f"{group_name:<15} {result['n_channels']:^10} "
              f"{result['accuracy']:.4f}      {result['total_importance']:.4f}      "
              f"{result['training_time']:.1f}s     {efficiency:.4f}")
    
    # Find best performing configurations
    best_overall = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_efficiency = max(results.keys(), key=lambda k: results[k]['accuracy'] / results[k]['n_channels'])
    
    print(f"\n🏆 BEST RESULTS:")
    print(f"   Highest Accuracy: {best_overall} ({results[best_overall]['accuracy']:.4f})")
    print(f"   Best Efficiency:  {best_efficiency} ({results[best_efficiency]['accuracy'] / results[best_efficiency]['n_channels']:.4f} acc/channel)")
    
    # Channel tier analysis
    print(f"\n📈 CHANNEL TIER ANALYSIS:")
    print("-" * 50)
    tier_results = {
        'tier_1_top3': results['tier_1_top3'],
        'tier_2_mid3': results['tier_2_mid3'], 
        'tier_3_bottom3': results['tier_3_bottom3']
    }
    
    for tier_name, result in tier_results.items():
        improvement = result['accuracy'] - baseline_acc
        print(f"{tier_name:15s}: {result['accuracy']:.4f} ({improvement:+.4f} vs full)")
        print(f"                 Channels: {result['channels']}")
    
    # Save results
    save_channel_analysis_results(results, sorted_channels, total_time)
    
    print(f"\n✅ CHANNEL ANALYSIS COMPLETED!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return results

def prepare_channel_data_for_cnn(X, y, selected_channels, all_channels):
    """Prepare data subset based on selected channels"""
    channel_indices = [all_channels.index(ch) for ch in selected_channels if ch in all_channels]
    
    if not channel_indices:
        raise ValueError(f"❌ No valid channels found: {selected_channels}")
    
    X_subset = X[:, :, channel_indices]
    print(f"   Selected channels: {[all_channels[i] for i in channel_indices]}")
    
    return X_subset, y

def save_channel_analysis_results(results, sorted_channels, total_time):
    """Save channel analysis results"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results structure
    analysis_results = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset': '285_patients_channel_analysis',
            'total_time': total_time
        },
        'channel_ranking': [
            {'rank': i+1, 'channel': ch, 'importance': float(imp)}
            for i, (ch, imp) in enumerate(sorted_channels)
        ],
        'group_results': {}
    }
    
    # Process group results
    for group_name, result in results.items():
        analysis_results['group_results'][group_name] = {
            'accuracy': float(result['accuracy']),
            'n_channels': int(result['n_channels']),
            'channels': result['channels'],
            'training_time': float(result['training_time']),
            'total_importance': float(result['total_importance']),
            'efficiency': float(result['accuracy'] / result['n_channels'])
        }
    
    # Save results
    results_file = os.path.join(CONFIG['output_dir'], f'channel_analysis_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"💾 Channel analysis saved: {results_file}")

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()