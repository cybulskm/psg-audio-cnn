# filepath: PSG-Audio/PSG-Audio/config/config.py
DATA_PATH = "/raid/userdata/cybulskm/ThesisProj/100_patients_processed.pkl"
OUTPUT_DIR = "bin"
CHECKPOINT_FILE = "bin/experiment_checkpoint.json"
RESULTS_FILE = "bin/cnn_feature_selection_results.csv"
N_RUNS = 5
CHUNK_SIZE = 5000
N_PROCESSES = 4
CHANNELS = [
    "Leg 2", "Leg 1", "EEG C3-A2", "EEG C4-A1", 
    "EMG Chin", "EEG A1-A2", "EOG LOC-A2", 
    "EOG ROC-A2", "ECG I"
]
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
CLASS_WEIGHT_THRESHOLD = 3