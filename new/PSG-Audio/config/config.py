import os
import multiprocessing as mp

# FORCE CPU USAGE - CRITICAL FOR GPU-LESS SYSTEMS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CONFIG = {
    # Data configuration
    'data_path': "/raid/userdata/cybulskm/ThesisProj/100_patients_processed.pkl",  # Use 100-patient dataset
    'output_dir': "results",
    'channels': ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", 
                 "EMG Chin", "Leg 1", "Leg 2", "ECG I"],
    
    # Experiment configuration
    'test_size': 0.25,
    'validation_split': 0.2,
    'random_state': 42,
    'max_segments': None,  # Use all available segments
    'exclude_classes': ['MixedApnea'],  # Exclude MixedApnea for 3-class problem
    
    # Random Forest configuration
    'rf_config': {
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_split': 5,
        'min_samples_leaf': 4,
        'max_features': 'log2',
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    },
    
    # CNN configuration
    'cnn_config': {
        'learning_rate': 0.0003,
        'batch_size': 32,
        'epochs': 150,
        'patience': 25,
        'reduce_lr_patience': 12,
        'min_lr': 1e-6,
        'dropout_rates': [0.2, 0.25, 0.3, 0.4, 0.3, 0.2]
    },
    
    # Hardware configuration
    'n_processes': min(8, mp.cpu_count() - 4),
    'memory_limit_gb': 8
}