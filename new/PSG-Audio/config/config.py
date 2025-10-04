import os
import multiprocessing as mp

# FORCE CPU USAGE - CRITICAL FOR GPU-LESS SYSTEMS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

CONFIG = {
    # Data configuration
    'data_path': "/raid/userdata/cybulskm/ThesisProj/285_patients_processed.pkl",  # Use 285-patient dataset
    'output_dir': "results",
    'channels': ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", 
                 "EMG Chin", "Leg 1", "Leg 2", "ECG I"],
    
    # Experiment configuration
    'test_size': 0.25,
    'validation_split': 0.2,
    'random_state': 42,
    'max_segments': None,  # Use all available segments
    
    # Random Forest configuration
    'rf_config': {
    'n_estimators': 200,  # Consider increasing for large datasets
    'max_depth': 25,  # Might be too deep - could overfit
    'min_samples_split': 5,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',  # Often better than 'log2' for large datasets
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced',
    'bootstrap': True,
    'verbose': 1  # Add progress monitoring
    },
    
    'cnn_config': {
    'learning_rate': 0.0003,  # Good for stability
    'batch_size': 32,  # Consider increasing to 64-128 if memory allows
    'epochs': 150,  # Reasonable with early stopping
    'patience': 25,  # Could be reduced to 15-20
    'reduce_lr_patience': 12,
    'min_lr': 1e-6,
    'dropout_rates': [0.2, 0.25, 0.3, 0.4, 0.3, 0.2],  # Good regularization
    'early_stopping_metric': 'val_loss'  # Explicitly define
    }
}