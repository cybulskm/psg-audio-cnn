import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from src.data_loader import load_data_streaming, filter_classes
from src.random_forest import get_feature_importance, select_top_features, convert_feature_names_to_channels
from src.cnn import train_and_evaluate_cnn
from config.config import CONFIG

def setup_cpu_environment():
    """Setup CPU-only environment (same as your original)"""
    print("=" * 70)
    print("CONFIGURED FOR CPU-ONLY EXECUTION")
    print("=" * 70)
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available devices: {[d.device_type for d in tf.config.get_visible_devices()]}")
    print("=" * 70)

def main():
    """Main function to orchestrate the workflow"""
    setup_cpu_environment()
    
    print("🚀 PSG-AUDIO: RANDOM FOREST → CNN PIPELINE")
    print("=" * 70)
    print(f"Using data: {CONFIG['data_path']}")
    print(f"Channels: {len(CONFIG['channels'])}")
    print(f"Excluding classes: {CONFIG['exclude_classes']}")
    print("=" * 70)

    # 1. Load and filter data
    print("\n📂 STEP 1: LOADING DATA")
    print("-" * 40)
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], CONFIG['max_segments'])
    
    # Filter out excluded classes (e.g., MixedApnea)
    X, y = filter_classes(X, y, CONFIG['exclude_classes'])
    
    # Validate data quality
    labels, y_encoded = np.unique(y, return_inverse=True)
    print(f"Labels found: {labels}")
    print(f"Final data shape: {X.shape}")
    
    if len(labels) < 2:
        raise ValueError("Need at least 2 classes for classification!")
    
    # Convert labels to categorical
    y_cat = to_categorical(y_encoded)
    print(f"Categorical labels shape: {y_cat.shape}")

    # 2. Get feature importance from Random Forest
    print("\n🌲 STEP 2: RANDOM FOREST FEATURE SELECTION")
    print("-" * 50)
    feature_importance = get_feature_importance(X, y_cat, CONFIG['channels'])

    # 3. Select top 25% and top 50% features
    print("\n📊 STEP 3: FEATURE SELECTION")
    print("-" * 40)
    
    # Get top features
    top_25_percent_features = select_top_features(feature_importance, percentage=0.25)
    top_50_percent_features = select_top_features(feature_importance, percentage=0.50)
    
    # Convert feature names back to channel names for CNN
    top_25_channels = convert_feature_names_to_channels(top_25_percent_features, CONFIG['channels'])
    top_50_channels = convert_feature_names_to_channels(top_50_percent_features, CONFIG['channels'])
    
    print(f"\nTop 25% features mapped to {len(top_25_channels)} channels: {top_25_channels}")
    print(f"Top 50% features mapped to {len(top_50_channels)} channels: {top_50_channels}")

    # 4. Prepare datasets for CNN
    print("\n🔧 STEP 4: PREPARING CNN DATASETS")
    print("-" * 40)
    
    X_25, y_25 = prepare_data_for_cnn(X, y_cat, top_25_channels, CONFIG['channels'])
    X_50, y_50 = prepare_data_for_cnn(X, y_cat, top_50_channels, CONFIG['channels'])
    X_all = X  # Full dataset for comparison
    
    print(f"Top 25% dataset shape: {X_25.shape}")
    print(f"Top 50% dataset shape: {X_50.shape}")
    print(f"Full dataset shape: {X_all.shape}")

    # 5. Train and evaluate CNN on different feature sets
    print("\n🤖 STEP 5: CNN TRAINING AND EVALUATION")
    print("-" * 50)
    
    results = {}
    
    # CNN with top 25% features
    print(f"\n{'='*20} CNN WITH TOP 25% FEATURES {'='*20}")
    acc_25, model_25, history_25 = train_and_evaluate_cnn(
        X_25, y_25, 
        f"(Top 25% Features - {len(top_25_channels)} channels)"
    )
    results['top_25_percent'] = {
        'accuracy': acc_25,
        'n_channels': len(top_25_channels),
        'channels': top_25_channels
    }
    
    # CNN with top 50% features
    print(f"\n{'='*20} CNN WITH TOP 50% FEATURES {'='*20}")
    acc_50, model_50, history_50 = train_and_evaluate_cnn(
        X_50, y_50,
        f"(Top 50% Features - {len(top_50_channels)} channels)"
    )
    results['top_50_percent'] = {
        'accuracy': acc_50,
        'n_channels': len(top_50_channels),
        'channels': top_50_channels
    }
    
    # CNN with all features (baseline)
    print(f"\n{'='*20} CNN WITH ALL FEATURES (BASELINE) {'='*20}")
    acc_all, model_all, history_all = train_and_evaluate_cnn(
        X_all, y_cat,
        f"(All Features - {len(CONFIG['channels'])} channels)"
    )
    results['all_features'] = {
        'accuracy': acc_all,
        'n_channels': len(CONFIG['channels']),
        'channels': CONFIG['channels']
    }

    # 6. Final comparison and results
    print("\n" + "="*70)
    print("🎯 FINAL RESULTS COMPARISON")
    print("="*70)
    
    print(f"{'Feature Set':<20} {'Channels':<10} {'Accuracy':<12} {'Improvement':<12}")
    print("-" * 60)
    
    baseline_acc = results['all_features']['accuracy']
    
    for key, result in results.items():
        improvement = result['accuracy'] - baseline_acc
        improvement_str = f"{improvement:+.4f}" if key != 'all_features' else "baseline"
        
        print(f"{key.replace('_', ' ').title():<20} "
              f"{result['n_channels']:<10} "
              f"{result['accuracy']:.4f}      "
              f"{improvement_str:<12}")
    
    # Determine best approach
    best_key = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_key]
    
    print(f"\n🏆 BEST PERFORMANCE: {best_key.replace('_', ' ').title()}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Channels: {best_result['n_channels']}")
    print(f"   Selected channels: {best_result['channels']}")
    
    # Save results
    save_results(results, feature_importance)
    
    print(f"\n✅ Pipeline completed successfully!")
    return results

def prepare_data_for_cnn(X, y, selected_channels, all_channels):
    """Prepare data for CNN based on selected channels"""
    channel_indices = [all_channels.index(ch) for ch in selected_channels if ch in all_channels]
    
    if not channel_indices:
        raise ValueError(f"No valid channels found in selection: {selected_channels}")
    
    X_subset = X[:, :, channel_indices]
    print(f"Selected {len(channel_indices)} channels: {[all_channels[i] for i in channel_indices]}")
    
    return X_subset, y

def save_results(results, feature_importance):
    """Save results to files"""
    import json
    from datetime import datetime
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results summary
    results_file = os.path.join(CONFIG['output_dir'], f"pipeline_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save feature importance
    feature_file = os.path.join(CONFIG['output_dir'], f"feature_importance_{timestamp}.txt")
    with open(feature_file, 'w') as f:
        f.write("FEATURE IMPORTANCE RANKING\n")
        f.write("=" * 50 + "\n")
        for i, (feature, importance) in enumerate(feature_importance, 1):
            f.write(f"{i:3d}. {feature:<40} {importance:.6f}\n")
    
    print(f"\n💾 Results saved:")
    print(f"   Summary: {results_file}")
    print(f"   Features: {feature_file}")

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()