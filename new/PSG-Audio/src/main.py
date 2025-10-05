import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
import psutil
from datetime import datetime

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import modules
from data_loader import load_data_streaming, filter_classes, validate_data_quality
from random_forest import get_feature_importance, select_top_features, convert_feature_names_to_channels
from cnn import train_and_evaluate_cnn
from config.config import CONFIG

def monitor_system_resources():
    """Monitor system resources if enabled"""
    if CONFIG['monitoring']['log_memory_usage']:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"💻 System Status:")
        print(f"   Memory: {memory.used/1e9:.1f}GB / {memory.total/1e9:.1f}GB ({memory.percent:.1f}%)")
        print(f"   CPU Usage: {cpu_percent:.1f}%")
        print(f"   Available Memory: {memory.available/1e9:.1f}GB")

def setup_optimized_environment():
    """Setup optimized environment for 1TB RAM system"""
    print("=" * 80)
    print("🚀 1TB RAM SYSTEM - MAXIMUM PERFORMANCE CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: 285-patient dataset ({CONFIG['data_path']})")
    print(f"Target memory usage: {CONFIG['hardware']['max_memory_gb']}GB")
    print(f"CPU cores available: {CONFIG['hardware']['n_processes']}")
    print("=" * 80)
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    # Configure TensorFlow for CPU optimization
    tf.config.threading.set_intra_op_parallelism_threads(CONFIG['hardware']['n_processes'])
    tf.config.threading.set_inter_op_parallelism_threads(CONFIG['hardware']['n_processes'])
    
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Available devices: {[d.device_type for d in tf.config.get_visible_devices()]}")
    print(f"CPU threads: intra={CONFIG['hardware']['n_processes']}, inter={CONFIG['hardware']['n_processes']}")
    
    monitor_system_resources()
    print("=" * 80)

def create_enhanced_output_directory():
    """Create enhanced output directory structure"""
    base_dir = CONFIG['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dirs_to_create = [
        base_dir,
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'tensorboard'),
        os.path.join(base_dir, 'plots'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, f'experiment_{timestamp}')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    return timestamp

def main():
    """Main orchestration function with full optimization"""
    pipeline_start = time.time()
    setup_optimized_environment()
    
    print("🧬 PSG-AUDIO: ADVANCED ML PIPELINE")
    print("🔬 285-PATIENT DATASET - 1TB RAM OPTIMIZED")
    print("=" * 80)
    
    # Create output directories
    experiment_timestamp = create_enhanced_output_directory()
    
    # Configuration summary
    print("📋 PIPELINE CONFIGURATION:")
    print("-" * 50)
    print(f"Data loading: {'Full in-memory' if CONFIG['data_loading']['load_full_in_memory'] else 'Streaming'}")
    print(f"RF trees: {CONFIG['rf_config']['n_estimators']}")
    print(f"RF max depth: {CONFIG['rf_config']['max_depth']}")
    print(f"CNN epochs: {CONFIG['cnn_config']['epochs']}")
    print(f"CNN batch size: {CONFIG['cnn_config']['batch_size']}")
    print(f"Cross-validation: {CONFIG['training']['cross_validation_folds']} folds")
    print(f"Advanced features: {CONFIG['training']['use_advanced_augmentation']}")
    print("-" * 50)

    # STEP 1: Advanced Data Loading
    print("\n📂 STEP 1: OPTIMIZED DATA LOADING")
    print("-" * 50)
    
    step_start = time.time()
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], CONFIG['max_segments'])
    
    # Enhanced data validation
    labels, y_encoded = np.unique(y, return_inverse=True)
    print(f"✅ Data loaded successfully:")
    print(f"   Shape: {X.shape}")
    print(f"   Labels: {labels}")
    print(f"   Memory: {X.nbytes / 1e9:.2f} GB")
    
    imbalance_ratio = validate_data_quality(X, y_encoded)
    step_time = time.time() - step_start
    print(f"   Loading time: {step_time:.1f}s")
    
    if len(labels) < 2:
        raise ValueError("❌ Need at least 2 classes for classification!")
    
    # Convert to categorical
    y_cat = to_categorical(y_encoded)
    print(f"   Categorical shape: {y_cat.shape}")
    
    monitor_system_resources()

    # STEP 2: Advanced Random Forest Feature Selection
    print("\n🌲 STEP 2: OPTIMIZED RANDOM FOREST ANALYSIS")
    print("-" * 50)
    
    step_start = time.time()
    feature_importance = get_feature_importance(X, y_cat, CONFIG['channels'])
    step_time = time.time() - step_start
    print(f"✅ Random Forest analysis completed in {step_time:.1f}s")

    # STEP 3: Multi-tier Feature Selection
    print("\n📊 STEP 3: MULTI-TIER FEATURE SELECTION")
    print("-" * 50)
    
    # Multiple feature selection percentages for comprehensive analysis
    feature_sets = {
        'top_10_percent': 0.10,
        'top_25_percent': 0.25,
        'top_50_percent': 0.50,
        'top_75_percent': 0.75
    }
    
    selected_feature_sets = {}
    
    for set_name, percentage in feature_sets.items():
        features = select_top_features(feature_importance, percentage=percentage)
        channels = convert_feature_names_to_channels(features, CONFIG['channels'])
        selected_feature_sets[set_name] = {
            'features': features,
            'channels': channels,
            'n_channels': len(channels),
            'percentage': percentage
        }
        print(f"✅ {set_name}: {len(features)} features → {len(channels)} channels")

    # STEP 4: Advanced Dataset Preparation
    print("\n🔧 STEP 4: ADVANCED DATASET PREPARATION")
    print("-" * 50)
    
    datasets = {}
    
    for set_name, feature_set in selected_feature_sets.items():
        X_subset, y_subset = prepare_optimized_data_for_cnn(
            X, y_cat, feature_set['channels'], CONFIG['channels']
        )
        datasets[set_name] = {
            'X': X_subset,
            'y': y_subset,
            'channels': feature_set['channels'],
            'n_channels': feature_set['n_channels']
        }
        print(f"   {set_name}: {X_subset.shape}")
    
    # Add full dataset
    datasets['full_dataset'] = {
        'X': X,
        'y': y_cat,
        'channels': CONFIG['channels'],
        'n_channels': len(CONFIG['channels'])
    }
    print(f"   full_dataset: {X.shape}")
    
    monitor_system_resources()

    # STEP 5: Comprehensive CNN Training & Evaluation
    print("\n🤖 STEP 5: COMPREHENSIVE CNN TRAINING")
    print("-" * 50)
    
    results = {}
    training_order = ['top_10_percent', 'top_25_percent', 'top_50_percent', 'top_75_percent', 'full_dataset']
    
    for i, dataset_name in enumerate(training_order, 1):
        dataset = datasets[dataset_name]
        
        print(f"\n{'='*20} CNN {i}/5: {dataset_name.upper()} {'='*20}")
        print(f"Channels: {dataset['n_channels']}")
        print(f"Selected: {dataset['channels']}")
        
        step_start = time.time()
        
        # Train and evaluate
        test_acc, model, history = train_and_evaluate_cnn(
            dataset['X'], dataset['y'],
            f"{dataset_name} ({dataset['n_channels']} channels)"
        )
        
        step_time = time.time() - step_start
        
        results[dataset_name] = {
            'accuracy': test_acc,
            'n_channels': dataset['n_channels'],
            'channels': dataset['channels'],
            'training_time': step_time,
            'history': history.history if history else None
        }
        
        print(f"✅ {dataset_name} completed: {test_acc:.4f} accuracy ({step_time:.1f}s)")
        monitor_system_resources()
        
        # Save intermediate results
        save_intermediate_results(results, experiment_timestamp, dataset_name)

    # STEP 6: Comprehensive Analysis & Reporting
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    total_time = time.time() - pipeline_start
    
    # Performance comparison table
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Dataset':<15} {'Channels':<10} {'Accuracy':<12} {'Time':<10} {'Improvement':<12}")
    print("-" * 80)
    
    baseline_acc = results['full_dataset']['accuracy']
    
    for dataset_name in training_order:
        result = results[dataset_name]
        improvement = result['accuracy'] - baseline_acc
        improvement_str = f"{improvement:+.4f}" if dataset_name != 'full_dataset' else "baseline"
        
        print(f"{dataset_name:<15} {result['n_channels']:<10} "
              f"{result['accuracy']:.4f}      {result['training_time']:.1f}s     "
              f"{improvement_str:<12}")
    
    # Find best performing model
    best_dataset = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_result = results[best_dataset]
    
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   Model: {best_dataset}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   Channels: {best_result['n_channels']}")
    print(f"   Improvement over baseline: {best_result['accuracy'] - baseline_acc:+.4f}")
    print(f"   Training time: {best_result['training_time']:.1f}s")
    
    # Efficiency analysis
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    print("-" * 50)
    for dataset_name in ['top_10_percent', 'top_25_percent', 'top_50_percent']:
        result = results[dataset_name]
        baseline_result = results['full_dataset']
        
        acc_ratio = result['accuracy'] / baseline_result['accuracy']
        channel_ratio = result['n_channels'] / baseline_result['n_channels']
        time_ratio = result['training_time'] / baseline_result['training_time']
        
        efficiency = acc_ratio / channel_ratio  # Accuracy per channel
        
        print(f"{dataset_name}:")
        print(f"   Accuracy efficiency: {efficiency:.3f} (acc/channel ratio)")
        print(f"   Time efficiency: {time_ratio:.3f} (relative to full)")
        print(f"   Channel reduction: {(1-channel_ratio)*100:.1f}%")
    
    # Save comprehensive results
    save_comprehensive_results(results, feature_importance, experiment_timestamp, total_time)
    
    print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved with timestamp: {experiment_timestamp}")
    
    return results, experiment_timestamp

def prepare_optimized_data_for_cnn(X, y, selected_channels, all_channels):
    """Prepare optimized data subsets for CNN training"""
    channel_indices = [all_channels.index(ch) for ch in selected_channels if ch in all_channels]
    
    if not channel_indices:
        raise ValueError(f"❌ No valid channels found in selection: {selected_channels}")
    
    X_subset = X[:, :, channel_indices]
    
    # Memory optimization
    if CONFIG['data_loading']['prefetch_to_device']:
        X_subset = X_subset.astype(np.float32)
    
    print(f"   Selected {len(channel_indices)} channels: {[all_channels[i] for i in channel_indices]}")
    print(f"   Subset memory: {X_subset.nbytes / 1e9:.2f} GB")
    
    return X_subset, y

def save_intermediate_results(results, timestamp, current_dataset):
    """Save intermediate results during training"""
    import json
    
    filename = os.path.join(CONFIG['output_dir'], f'intermediate_results_{timestamp}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {
            'accuracy': float(value['accuracy']),
            'n_channels': int(value['n_channels']),
            'channels': value['channels'],
            'training_time': float(value['training_time'])
        }
    
    with open(filename, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'completed_datasets': list(results.keys()),
            'current_dataset': current_dataset,
            'results': serializable_results
        }, f, indent=2)

def save_comprehensive_results(results, feature_importance, timestamp, total_time):
    """Save comprehensive results with enhanced metadata"""
    import json
    
    # Enhanced results structure
    comprehensive_results = {
        'experiment_metadata': {
            'timestamp': timestamp,
            'dataset': '285_patients',
            'system_config': {
                'total_memory_gb': psutil.virtual_memory().total / 1e9,
                'cpu_count': psutil.cpu_count(),
                'optimization_level': '1TB_RAM_System'
            },
            'pipeline_config': CONFIG,
            'total_execution_time': total_time
        },
        'feature_analysis': {
            'total_features': len(feature_importance),
            'top_10_features': [(feat, float(imp)) for feat, imp in feature_importance[:10]],
            'feature_importance_summary': {
                'max_importance': float(max(imp for _, imp in feature_importance)),
                'min_importance': float(min(imp for _, imp in feature_importance)),
                'mean_importance': float(np.mean([imp for _, imp in feature_importance]))
            }
        },
        'model_results': {}
    }
    
    # Process results for serialization
    for dataset_name, result in results.items():
        comprehensive_results['model_results'][dataset_name] = {
            'accuracy': float(result['accuracy']),
            'n_channels': int(result['n_channels']),
            'channels': result['channels'],
            'training_time_seconds': float(result['training_time']),
            'training_history': result.get('history', {})
        }
   