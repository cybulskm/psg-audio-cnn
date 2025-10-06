import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import random
import time
import json

# Add parent directory to path for config imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Import fixes
try:
    from src.data_loader import load_data_streaming, validate_data_quality
    from src.cnn import train_and_evaluate_cnn
except ImportError:
    from data_loader import load_data_streaming, validate_data_quality
    from cnn import train_and_evaluate_cnn

from config.config import CONFIG

# Remove previous CPU-only forcing. Let cnn module configure GPUs.
CHANNEL_RANKING = [
    ('ECG I', 0.128733),
    ('EEG C4-A1', 0.105451),
    ('EOG ROC-A2', 0.105369),
    ('EEG A1-A2', 0.105110),
    ('EOG LOC-A2', 0.102342),
    ('EEG C3-A2', 0.100272),
    ('Leg 2', 0.096922),
    ('Leg 1', 0.096168),
    ('EMG Chin', 0.092785)
]

def setup_cpu_environment():
    """Setup CPU-only environment"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print("=" * 80)
    print("🧪 CNN FEATURE SELECTION TEST (5 RUNS EACH)")
    print("=" * 80)
    print("Testing: Top 25%, Top 50%, Random selection")
    print("Runs per configuration: 5")
    print("=" * 80)
    
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    
    print(f"TensorFlow: {tf.__version__} (CPU-only)")
    print("=" * 80)

def create_channel_groups():
    """Create channel groups based on importance ranking"""
    all_channels = [ch for ch, _ in CHANNEL_RANKING]
    
    # Top 25% (2.25 ≈ 2 channels, but we'll use 3 for better representation)
    top_25_percent = all_channels[:3]  # Top 3 channels (33%)
    
    # Top 50% (4.5 ≈ 4-5 channels)
    top_50_percent = all_channels[:5]  # Top 5 channels (56%)
    
    # Random assortment (5 channels selected randomly each time)
    def get_random_channels():
        return random.sample(all_channels, 5)
    
    groups = {
        'top_25_percent': {
            'channels': top_25_percent,
            'description': f"Top 25% ({len(top_25_percent)} channels)",
            'fixed': True
        },
        'top_50_percent': {
            'channels': top_50_percent, 
            'description': f"Top 50% ({len(top_50_percent)} channels)",
            'fixed': True
        },
        'random_selection': {
            'channels': get_random_channels,  # Function to generate random each time
            'description': "Random selection (5 channels)",
            'fixed': False
        }
    }
    
    print(f"\n📊 TEST CONFIGURATIONS:")
    print("-" * 60)
    print(f"Top 25%: {top_25_percent}")
    print(f"Top 50%: {top_50_percent}")
    print(f"Random:  Generated randomly each run (5 channels)")
    print("-" * 60)
    
    return groups

def prepare_channel_data(X, y, selected_channels, all_available_channels):
    """Prepare data subset for selected channels"""
    # Map selected channels to indices
    channel_indices = []
    for ch in selected_channels:
        if ch in all_available_channels:
            channel_indices.append(all_available_channels.index(ch))
        else:
            # Handle slight name mismatches
            for i, available_ch in enumerate(all_available_channels):
                if ch.replace(' ', '').replace('-', '') in available_ch.replace(' ', '').replace('-', ''):
                    channel_indices.append(i)
                    break
    
    if not channel_indices:
        raise ValueError(f"❌ No valid channels found for: {selected_channels}")
    
    X_subset = X[:, :, channel_indices]
    mapped_channels = [all_available_channels[i] for i in channel_indices]
    
    print(f"   Selected: {selected_channels}")
    print(f"   Mapped to: {mapped_channels}")
    print(f"   Data shape: {X_subset.shape}")
    
    return X_subset, y, mapped_channels

def run_multiple_tests(X, y, group_name, group_config, all_channels, n_runs=5):
    """Run multiple CNN tests for a channel configuration"""
    print(f"\n🔬 TESTING: {group_name.upper()}")
    print("=" * 60)
    print(f"Description: {group_config['description']}")
    print(f"Number of runs: {n_runs}")
    
    results = []
    
    for run in range(1, n_runs + 1):
        print(f"\n--- RUN {run}/{n_runs} ---")
        
        # Get channels for this run
        if group_config['fixed']:
            selected_channels = group_config['channels']
        else:
            selected_channels = group_config['channels']()  # Call function for random
        
        run_start = time.time()
        
        try:
            # Prepare data
            X_subset, y_subset, mapped_channels = prepare_channel_data(
                X, y, selected_channels, all_channels
            )
            
            # Train CNN
            test_acc, model, history = train_and_evaluate_cnn(
                X_subset, y_subset,
                f"{group_name} Run {run} ({len(mapped_channels)} channels)"
            )
            
            run_time = time.time() - run_start
            
            run_result = {
                'run': run,
                'accuracy': test_acc,
                'channels': mapped_channels,
                'n_channels': len(mapped_channels),
                'training_time': run_time,
                'success': True
            }
            
            print(f"✅ Run {run}: {test_acc:.4f} accuracy ({run_time:.1f}s)")
            
        except Exception as e:
            run_result = {
                'run': run,
                'accuracy': 0.0,
                'channels': selected_channels,
                'n_channels': len(selected_channels),
                'training_time': 0.0,
                'success': False,
                'error': str(e)
            }
            print(f"❌ Run {run}: Failed - {e}")
        
        results.append(run_result)
    
    return results

def analyze_results(all_results):
    """Analyze and summarize test results"""
    print(f"\n" + "="*80)
    print("📈 TEST RESULTS ANALYSIS")
    print("="*80)
    
    summary = {}
    
    for group_name, results in all_results.items():
        # Filter successful runs
        successful_runs = [r for r in results if r['success']]
        
        if successful_runs:
            accuracies = [r['accuracy'] for r in successful_runs]
            times = [r['training_time'] for r in successful_runs]
            
            summary[group_name] = {
                'successful_runs': len(successful_runs),
                'total_runs': len(results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies),
                'mean_time': np.mean(times),
                'n_channels': successful_runs[0]['n_channels']
            }
        else:
            summary[group_name] = {
                'successful_runs': 0,
                'total_runs': len(results),
                'mean_accuracy': 0.0,
                'std_accuracy': 0.0,
                'min_accuracy': 0.0,
                'max_accuracy': 0.0,
                'mean_time': 0.0,
                'n_channels': 0
            }
    
    # Print summary table
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print("-" * 100)
    print(f"{'Group':<15} {'Success':<8} {'Channels':<9} {'Mean Acc':<10} {'Std':<8} {'Min':<8} {'Max':<8} {'Time':<8}")
    print("-" * 100)
    
    for group_name, stats in summary.items():
        success_rate = f"{stats['successful_runs']}/{stats['total_runs']}"
        print(f"{group_name:<15} {success_rate:<8} {stats['n_channels']:<9} "
              f"{stats['mean_accuracy']:.4f}    {stats['std_accuracy']:.4f}  "
              f"{stats['min_accuracy']:.4f}  {stats['max_accuracy']:.4f}  {stats['mean_time']:.1f}s")
    
    # Statistical comparisons
    print(f"\n📈 STATISTICAL ANALYSIS:")
    print("-" * 50)
    
    # Best mean performance
    best_group = max(summary.keys(), key=lambda k: summary[k]['mean_accuracy'])
    print(f"Best mean accuracy: {best_group} ({summary[best_group]['mean_accuracy']:.4f})")
    
    # Most consistent (lowest std)
    consistent_groups = [k for k, v in summary.items() if v['successful_runs'] > 0]
    if consistent_groups:
        most_consistent = min(consistent_groups, key=lambda k: summary[k]['std_accuracy'])
        print(f"Most consistent: {most_consistent} (std: {summary[most_consistent]['std_accuracy']:.4f})")
    
    # Efficiency (accuracy per channel)
    efficiency_analysis = {}
    for group_name, stats in summary.items():
        if stats['n_channels'] > 0:
            efficiency_analysis[group_name] = stats['mean_accuracy'] / stats['n_channels']
    
    if efficiency_analysis:
        most_efficient = max(efficiency_analysis.keys(), key=lambda k: efficiency_analysis[k])
        print(f"Most efficient: {most_efficient} ({efficiency_analysis[most_efficient]:.4f} acc/channel)")
    
    return summary

def save_test_results(all_results, summary, total_time):
    """Save comprehensive test results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results for JSON serialization
    json_results = {
        'experiment_info': {
            'timestamp': timestamp,
            'experiment_type': 'feature_selection_test',
            'runs_per_group': 5,
            'total_time': total_time,
            'channel_ranking': [{'channel': ch, 'importance': imp} for ch, imp in CHANNEL_RANKING]
        },
        'detailed_results': {},
        'summary_statistics': {}
    }
    
    # Add detailed results
    for group_name, results in all_results.items():
        json_results['detailed_results'][group_name] = []
        for result in results:
            json_result = {
                'run': result['run'],
                'accuracy': float(result['accuracy']),
                'channels': result['channels'],
                'n_channels': result['n_channels'],
                'training_time': float(result['training_time']),
                'success': result['success']
            }
            if 'error' in result:
                json_result['error'] = result['error']
            json_results['detailed_results'][group_name].append(json_result)
    
    # Add summary statistics
    for group_name, stats in summary.items():
        json_results['summary_statistics'][group_name] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in stats.items()
        }
    
    # Save to file
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    results_file = os.path.join(CONFIG['output_dir'], f'feature_selection_test_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Test results saved: {results_file}")
    return results_file

def main():
    """Main test function"""
    test_start = time.time()
    setup_cpu_environment()
    
    print("🧪 CNN FEATURE SELECTION COMPREHENSIVE TEST")
    print("=" * 80)

    # Load data
    print("\n📂 LOADING DATA...")
    X, y = load_data_streaming(CONFIG['data_path'], CONFIG['channels'], CONFIG['max_segments'])
    labels, y_encoded = np.unique(y, return_inverse=True)
    y_cat = to_categorical(y_encoded)
    
    print(f"✅ Data loaded: {X.shape}, Labels: {labels}")
    validate_data_quality(X, y_encoded)

    # Create test configurations
    channel_groups = create_channel_groups()
    
    # Run tests for each group
    all_results = {}
    
    for group_name, group_config in channel_groups.items():
        group_results = run_multiple_tests(
            X, y_cat, group_name, group_config, CONFIG['channels'], n_runs=5
        )
        all_results[group_name] = group_results

    # Analyze results
    summary = analyze_results(all_results)
    
    # Save results
    total_time = time.time() - test_start
    results_file = save_test_results(all_results, summary, total_time)
    
    print(f"\n✅ COMPREHENSIVE TEST COMPLETED!")
    print(f"Total test time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Results file: {results_file}")
    
    return all_results, summary

if __name__ == "__main__":
    try:
        random.seed(42)  # For reproducible random selections
        np.random.seed(42)
        
        all_results, summary = main()
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()