import os
import multiprocessing as mp

# FORCE CPU USAGE - CRITICAL FOR GPU-LESS SYSTEMS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CONFIG = {
    # Data configuration
    'data_path': "/raid/userdata/cybulskm/ThesisProj/285_patients_processed.pkl",
    'output_dir': "results",
    'channels': ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", 
                 "EMG Chin", "Leg 1", "Leg 2", "ECG I"],
    
    # Experiment configuration
    'test_size': 0.25,
    'validation_split': 0.2,
    'random_state': 42,
    'max_segments': None,  # Use all available segments
    
    # Hardware optimization for 1TB RAM
    'hardware': {
        'n_processes': mp.cpu_count(),  # Use ALL cores
        'max_memory_gb': 900,  # Leave 100GB for OS/other processes
        'prefetch_factor': 4,
        'use_memory_mapping': False,  # Not needed with ample RAM
    },
    
    # Random Forest configuration - OPTIMIZED FOR LARGE RAM
    'rf_config': {
        'n_estimators': 500,  # Increased - more trees for better performance
        'max_depth': 30,  # Deeper trees to capture complex patterns
        'min_samples_split': 10,  # Slightly higher to prevent overfitting
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,  # Use all cores safely with 1TB RAM
        'class_weight': 'balanced',
        'bootstrap': True,
        'verbose': 2,
        'warm_start': False,  # No need with ample memory
        'oob_score': True,  # Use out-of-bag scoring
    },
    
    # CNN configuration - OPTIMIZED FOR PERFORMANCE
    'cnn_config': {
        'learning_rate': 0.001,  # Higher learning rate for faster convergence
        'batch_size': 256,  # Much larger batches for stability & speed
        'epochs': 100,  # Reduced since more data per batch
        'patience': 20,  # Slightly reduced patience
        'reduce_lr_patience': 10,
        'min_lr': 1e-7,
        'dropout_rates': [0.2, 0.3, 0.4, 0.5, 0.4, 0.3],  # Slightly more regularization
        'early_stopping_metric': 'val_loss',
        'use_advanced_optimizer': 'adamw',  # AdamW often better than Adam
        'weight_decay': 1e-4,
    },
    
    # Data loading optimization
    'data_loading': {
        'load_full_in_memory': True,  # Load entire 28GB dataset into RAM
        'shuffle_buffer_size': 100000,  # Large shuffle buffer
        'prefetch_to_device': True,
        'cache_dataset': True,
    },
    
    # Advanced training options
    'training': {
        'use_advanced_augmentation': True,
        'cross_validation_folds': 5,
        'save_checkpoints': True,
        'checkpoint_frequency': 10,
        'ensemble_models': True,  # Train multiple models for ensemble
    },
    
    # Monitoring and logging
    'monitoring': {
        'log_memory_usage': True,
        'profile_training': True,
        'tensorboard_logging': True,
        'save_training_curves': True,
    }
}