import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import time
from datetime import datetime
import json
import logging

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

# Setup logging
def setup_logging():
    """Setup logging to both file and console"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Setup file handler
    log_file = os.path.join('logs', 'experiment.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

def setup_optimized_environment():
    """Setup optimized environment"""
    logger.info("=" * 80)
    logger.info("🚀 PSG-AUDIO FEATURE SELECTION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Dataset: 285-patient dataset ({CONFIG['data_path']})")
    logger.info("=" * 80)
    
    import tensorflow as tf
    
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Detailed GPU check
    physical_gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    visible_devices = tf.config.get_visible_devices()
    
    logger.info(f"\n🔍 GPU STATUS:")
    logger.info(f"  Physical GPUs: {len(physical_gpus)}")
    logger.info(f"  Logical GPUs: {len(logical_gpus)}")
    logger.info(f"  Visible devices: {[d.device_type for d in visible_devices]}")
    
    if physical_gpus:
        for i, gpu in enumerate(physical_gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
        logger.info(f"✅ GPU IS AVAILABLE AND WILL BE USED FOR TRAINING")
    else:
        logger.info(f"❌ NO GPU DETECTED - WILL USE CPU ONLY")
    
    # Test GPU with a simple operation
    if physical_gpus:
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
            logger.info(f"✅ GPU test operation successful: GPU is functioning properly")
        except Exception as e:
            logger.warning(f"⚠️ GPU test failed: {e}")
    
    logger.info("=" * 80)

def extract_features_by_importance(X, feature_importance, channels, percentage):
    """Extract features based on importance percentage"""
    logger.info(f"\n🔍 Extracting top {percentage*100:.0f}% features")
    
    # Select top features
    top_features = select_top_features(feature_importance, percentage=percentage)
    
    logger.info(f"Selected {len(top_features)} features from {len(feature_importance)} total")
    
    # Convert feature names to channel indices
    selected_channels = convert_feature_names_to_channels(top_features, channels)
    
    logger.info(f"Channels represented: {len(selected_channels)}/{len(channels)}")
    logger.info(f"Selected channels: {selected_channels}")
    
    # Get channel indices
    channel_indices = [channels.index(ch) for ch in selected_channels if ch in channels]
    
    # Extract subset of data
    X_subset = X[:, :, channel_indices]
    
    logger.info(f"Data subset shape: {X_subset.shape}")
    
    return X_subset, selected_channels, top_features

def extract_random_features(X, feature_importance, channels, num_features):
    """Extract random features for baseline comparison"""
    logger.info(f"\n🎲 Extracting {num_features} random features")
    
    # Randomly select features
    all_features = [feat for feat, _ in feature_importance]
    np.random.seed(42)  # For reproducibility
    random_features = np.random.choice(all_features, size=num_features, replace=False).tolist()
    
    logger.info(f"Selected {len(random_features)} random features")
    
    # Convert to channels
    selected_channels = convert_feature_names_to_channels(random_features, channels)
    
    logger.info(f"Channels represented: {len(selected_channels)}/{len(channels)}")
    logger.info(f"Selected channels: {selected_channels}")
    
    # Get channel indices
    channel_indices = [channels.index(ch) for ch in selected_channels if ch in channels]
    
    # Extract subset
    X_subset = X[:, :, channel_indices]
    
    logger.info(f"Data subset shape: {X_subset.shape}")
    
    return X_subset, selected_channels, random_features

def run_cnn_experiment(X, y, name, num_runs=5):
    """Run CNN multiple times and collect statistics"""
    logger.info(f"\n{'='*80}")
    logger.info(f"🤖 RUNNING CNN EXPERIMENT: {name}")
    logger.info(f"{'='*80}")
    logger.info(f"Number of runs: {num_runs}")
    logger.info(f"Data shape: {X.shape}")
    
    results = []
    
    for run in range(1, num_runs + 1):
        logger.info(f"\n{'─'*60}")
        logger.info(f"Run {run}/{num_runs}")
        logger.info(f"{'─'*60}")
        
        run_start = time.time()
        
        try:
            test_acc, model, history = train_and_evaluate_cnn(
                X, y,
                f"{name} - Run {run}"
            )
            
            run_time = time.time() - run_start
            
            # Extract training history
            final_train_acc = history.history['accuracy'][-1] if 'accuracy' in history.history else 0
            final_val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
            best_val_acc = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0
            epochs_trained = len(history.history['accuracy']) if 'accuracy' in history.history else 0
            
            result = {
                'run': run,
                'test_accuracy': float(test_acc),
                'final_train_accuracy': float(final_train_acc),
                'final_val_accuracy': float(final_val_acc),
                'best_val_accuracy': float(best_val_acc),
                'epochs_trained': int(epochs_trained),
                'training_time': float(run_time)
            }
            
            results.append(result)
            
            logger.info(f"✅ Run {run} completed:")
            logger.info(f"   Test Accuracy: {test_acc:.4f}")
            logger.info(f"   Training Time: {run_time:.1f}s")
            
            # Clear session to free memory
            import tensorflow as tf
            tf.keras.backend.clear_session()
            
        except Exception as e:
            logger.error(f"❌ Run {run} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Calculate statistics
    if results:
        test_accs = [r['test_accuracy'] for r in results]
        train_times = [r['training_time'] for r in results]
        
        stats = {
            'mean_test_accuracy': float(np.mean(test_accs)),
            'std_test_accuracy': float(np.std(test_accs)),
            'min_test_accuracy': float(np.min(test_accs)),
            'max_test_accuracy': float(np.max(test_accs)),
            'mean_training_time': float(np.mean(train_times)),
            'total_training_time': float(np.sum(train_times))
        }
        
        logger.info(f"\n📊 {name} - Statistics across {len(results)} runs:")
        logger.info(f"   Mean Test Accuracy: {stats['mean_test_accuracy']:.4f} ± {stats['std_test_accuracy']:.4f}")
        logger.info(f"   Min/Max Accuracy: {stats['min_test_accuracy']:.4f} / {stats['max_test_accuracy']:.4f}")
        logger.info(f"   Mean Training Time: {stats['mean_training_time']:.1f}s")
        logger.info(f"   Total Training Time: {stats['total_training_time']:.1f}s")
        
        return results, stats
    else:
        logger.error(f"❌ No successful runs for {name}")
        return [], {}

def main():
    """Main experiment function"""
    experiment_start = time.time()
    setup_optimized_environment()
    
    logger.info("\n🧬 PSG-AUDIO: FEATURE SELECTION EXPERIMENT")
    logger.info("🔬 285-PATIENT DATASET")
    logger.info("=" * 80)
    
    # Experiment configuration
    NUM_RUNS = 5
    TOP_25_PERCENT = 0.25
    TOP_50_PERCENT = 0.50
    
    logger.info(f"\nExperiment Configuration:")
    logger.info(f"  Runs per experiment: {NUM_RUNS}")
    logger.info(f"  Feature selection percentages: {TOP_25_PERCENT*100:.0f}%, {TOP_50_PERCENT*100:.0f}%, Random")

    # STEP 1: Load Data
    logger.info("\n" + "="*80)
    logger.info("📂 STEP 1: DATA LOADING")
    logger.info("="*80)
    
    step_start = time.time()
    X, y, channels = load_data_streaming(
        CONFIG['data_path'], 
        channels=None,  # Auto-detect
        max_segments=CONFIG.get('max_segments') 
    )
    
    labels, y_encoded = np.unique(y, return_inverse=True)
    logger.info(f"✅ Data loaded: {X.shape}, Labels: {labels}")
    logger.info(f"✅ Channels detected: {channels}")
    
    imbalance_ratio = validate_data_quality(X, y_encoded)
    step_time = time.time() - step_start
    logger.info(f"   Loading time: {step_time:.1f}s")
    
    y_cat = to_categorical(y_encoded)

    # STEP 2: Random Forest Feature Importance
    logger.info("\n" + "="*80)
    logger.info("🌲 STEP 2: RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    
    step_start = time.time()
    feature_importance = get_feature_importance(X, y_cat, channels)
    rf_time = time.time() - step_start
    
    logger.info(f"✅ Random Forest completed in {rf_time:.1f}s")
    logger.info(f"   Total features analyzed: {len(feature_importance)}")
    
    # Log top 20 features
    logger.info("\n📊 Top 20 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:20], 1):
        logger.info(f"   {i:2d}. {feature:40s}: {importance:.6f}")

    # STEP 3: Prepare Feature Subsets
    logger.info("\n" + "="*80)
    logger.info("🔧 STEP 3: PREPARING FEATURE SUBSETS")
    logger.info("="*80)
    
    # Extract top 25% features
    X_top25, channels_top25, features_top25 = extract_features_by_importance(
        X, feature_importance, channels, TOP_25_PERCENT
    )
    
    # Extract top 50% features
    X_top50, channels_top50, features_top50 = extract_features_by_importance(
        X, feature_importance, channels, TOP_50_PERCENT
    )
    
    # Extract random features (same number as top 25%)
    num_random_features = len(features_top25)
    X_random, channels_random, features_random = extract_random_features(
        X, feature_importance, channels, num_random_features
    )
    
    # Prepare experiment datasets
    experiments = {
        'top_25_percent': {
            'X': X_top25,
            'y': y_cat,
            'channels': channels_top25,
            'features': features_top25,
            'description': f'Top {TOP_25_PERCENT*100:.0f}% features'
        },
        'top_50_percent': {
            'X': X_top50,
            'y': y_cat,
            'channels': channels_top50,
            'features': features_top50,
            'description': f'Top {TOP_50_PERCENT*100:.0f}% features'
        },
        'random_features': {
            'X': X_random,
            'y': y_cat,
            'channels': channels_random,
            'features': features_random,
            'description': f'Random {num_random_features} features (baseline)'
        }
    }

    # STEP 4: Run CNN Experiments
    logger.info("\n" + "="*80)
    logger.info("🤖 STEP 4: RUNNING CNN EXPERIMENTS")
    logger.info("="*80)
    
    all_results = {}
    
    for exp_name, exp_data in experiments.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment: {exp_name.upper()}")
        logger.info(f"Description: {exp_data['description']}")
        logger.info(f"Data shape: {exp_data['X'].shape}")
        logger.info(f"Channels ({len(exp_data['channels'])}): {exp_data['channels']}")
        logger.info(f"{'='*80}")
        
        runs, stats = run_cnn_experiment(
            exp_data['X'], 
            exp_data['y'], 
            exp_name,
            num_runs=NUM_RUNS
        )
        
        all_results[exp_name] = {
            'runs': runs,
            'statistics': stats,
            'data_shape': list(exp_data['X'].shape),
            'num_channels': len(exp_data['channels']),
            'channels': exp_data['channels'],
            'num_features': len(exp_data['features']),
            'features': exp_data['features'][:50]  # Store first 50 features to avoid huge JSON
        }

    # STEP 5: Compare Results
    logger.info("\n" + "="*80)
    logger.info("📊 STEP 5: EXPERIMENT COMPARISON")
    logger.info("="*80)
    
    # Create comparison table
    logger.info("\n📈 Performance Comparison:")
    logger.info("-" * 100)
    logger.info(f"{'Experiment':<20} {'Features':<10} {'Channels':<10} {'Mean Acc':<12} {'Std':<10} {'Min/Max':<15} {'Time (s)':<12}")
    logger.info("-" * 100)
    
    for exp_name in ['top_25_percent', 'top_50_percent', 'random_features']:
        if exp_name in all_results and all_results[exp_name]['statistics']:
            result = all_results[exp_name]
            stats = result['statistics']
            
            logger.info(
                f"{exp_name:<20} "
                f"{result['num_features']:<10} "
                f"{result['num_channels']:<10} "
                f"{stats['mean_test_accuracy']:.4f}      "
                f"±{stats['std_test_accuracy']:.4f}   "
                f"{stats['min_test_accuracy']:.4f}/{stats['max_test_accuracy']:.4f}    "
                f"{stats['mean_training_time']:.1f}"
            )
    
    logger.info("-" * 100)
    
    # Statistical comparison
    if all('statistics' in all_results[exp] for exp in experiments.keys()):
        logger.info("\n📊 Statistical Analysis:")
        
        top25_acc = all_results['top_25_percent']['statistics']['mean_test_accuracy']
        top50_acc = all_results['top_50_percent']['statistics']['mean_test_accuracy']
        random_acc = all_results['random_features']['statistics']['mean_test_accuracy']
        
        logger.info(f"   Top 25% vs Top 50%: {top25_acc - top50_acc:+.4f} accuracy difference")
        logger.info(f"   Top 25% vs Random: {top25_acc - random_acc:+.4f} accuracy difference")
        logger.info(f"   Top 50% vs Random: {top50_acc - random_acc:+.4f} accuracy difference")
        
        # Efficiency analysis
        top25_efficiency = top25_acc / all_results['top_25_percent']['num_features']
        top50_efficiency = top50_acc / all_results['top_50_percent']['num_features']
        random_efficiency = random_acc / all_results['random_features']['num_features']
        
        logger.info(f"\n⚡ Efficiency (Accuracy per Feature):")
        logger.info(f"   Top 25%: {top25_efficiency:.6f}")
        logger.info(f"   Top 50%: {top50_efficiency:.6f}")
        logger.info(f"   Random:  {random_efficiency:.6f}")

    # STEP 6: Save Results
    logger.info("\n" + "="*80)
    logger.info("💾 STEP 6: SAVING RESULTS")
    logger.info("="*80)
    
    total_time = time.time() - experiment_start
    
    # Create results directory
    os.makedirs(CONFIG.get('output_dir', 'results'), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive results to JSON
    experiment_results = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset': '285_patients_feature_selection',
            'total_time': float(total_time),
            'num_runs_per_experiment': NUM_RUNS,
            'feature_percentages': [TOP_25_PERCENT, TOP_50_PERCENT, 'random'],
            'classes': labels.tolist(),
            'total_samples': int(X.shape[0]),
            'total_channels': len(channels),
            'all_channels': channels
        },
        'random_forest': {
            'execution_time': float(rf_time),
            'total_features': len(feature_importance),
            'top_50_features': [
                {'rank': i+1, 'feature': feat, 'importance': float(imp)}
                for i, (feat, imp) in enumerate(feature_importance[:50])
            ]
        },
        'experiments': all_results
    }
    
    # Save to JSON
    results_file = os.path.join(
        CONFIG.get('output_dir', 'results'), 
        f'feature_selection_experiment_{timestamp}.json'
    )
    
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    logger.info(f"✅ Results saved to: {results_file}")
    
    # Also save a summary CSV
    summary_file = os.path.join(
        CONFIG.get('output_dir', 'results'),
        f'experiment_summary_{timestamp}.csv'
    )
    
    summary_data = []
    for exp_name, exp_result in all_results.items():
        if exp_result['statistics']:
            summary_data.append({
                'Experiment': exp_name,
                'Num_Features': exp_result['num_features'],
                'Num_Channels': exp_result['num_channels'],
                'Mean_Accuracy': exp_result['statistics']['mean_test_accuracy'],
                'Std_Accuracy': exp_result['statistics']['std_test_accuracy'],
                'Min_Accuracy': exp_result['statistics']['min_test_accuracy'],
                'Max_Accuracy': exp_result['statistics']['max_test_accuracy'],
                'Mean_Time': exp_result['statistics']['mean_training_time'],
                'Total_Time': exp_result['statistics']['total_training_time']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"✅ Summary saved to: {summary_file}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Total Experiment Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info(f"Logs saved to: logs/experiment.log")
    logger.info("="*80)
    
    return experiment_results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)