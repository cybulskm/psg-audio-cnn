import os
import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import csv
import pickle

relevant_channels = ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", "EMG Chin", "Leg 1", "Leg 2", "ECG I"]

def parse_annotations(rml_file_path):
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
    events = []
    for event in root.findall('.//ns:Event', namespace):
        event_type = event.get('Type')
        if "apnea" in event_type.lower():
            start_time = float(event.get('Start'))
            duration = float(event.get('Duration'))
            events.append((event_type, start_time, duration))
    print(f"Found {len(events)} events in {rml_file_path}")
    return events

def extract_features_from_edf(edf_file_path):
    edf_file = pyedflib.EdfReader(edf_file_path)
    signal_labels = edf_file.getSignalLabels()
    features = {channel: [] for channel in relevant_channels}
    sampling_rates = {}
    
    for channel in signal_labels:
        if channel in relevant_channels:
            try:
                signal_index = edf_file.getSignalLabels().index(channel)
                signal_data = edf_file.readSignal(signal_index)
                sampling_rate = edf_file.getSampleFrequency(signal_index)
                features[channel] = signal_data
                sampling_rates[channel] = sampling_rate
            except ValueError:
                print(f"Channel {channel} not found in the EDF file.")
                features[channel] = None
                sampling_rates[channel] = None
    
    edf_file.close()
    
    # Calculate min_length excluding channels with no data
    min_length = min(len(signal) for signal in features.values() if signal is not None and len(signal) > 0)
    print(f"Minimum length of signals: {min_length}")
    
    for channel in relevant_channels:
        if features[channel] is not None and len(features[channel]) > 0:
            features[channel] = features[channel][:min_length]
        else:
            features[channel] = [np.nan] * min_length
    
    features_df = pd.DataFrame(features)
    print(f"Extracted features from {edf_file_path}, DataFrame shape: {features_df.shape}")
    return features_df, sampling_rates

def preprocess_and_label(edf_file_path, annotations, remaining_annotations, sampling_rate=200):
    features_df, sampling_rates = extract_features_from_edf(edf_file_path)
    
    scaler = StandardScaler()
    for channel in features_df.columns:
        if features_df[channel].isnull().all():
            continue
        features_df[channel] = scaler.fit_transform(features_df[channel].values.reshape(-1, 1)).flatten()
    
    segments = []
    window_size = 60 * sampling_rate
    edf_duration = len(features_df) / sampling_rate
    print(f"EDF duration: {edf_duration} seconds, Sampling rate: {sampling_rate}, DataFrame length: {len(features_df)}")
    
    for event in annotations:
        label, start_time, _ = event
        print(f"Processing event {event} at {start_time}")

        if start_time >= edf_duration:
            remaining_annotations.append((label, start_time - edf_duration, _))
            continue
        
        start_idx = int(start_time * sampling_rate)
        end_idx = start_idx + window_size
        segment_data = {}
        for channel in features_df.columns:
            if channel in sampling_rates and sampling_rates[channel] is not None:
                channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                if len(channel_data) < window_size:
                    channel_data += [np.nan] * (window_size - len(channel_data))
                segment_data[channel] = channel_data
            else:
                segment_data[channel] = [np.nan] * window_size
        
        segment_data['Label'] = label
        segments.append(segment_data)
    
    print(f"Generated {len(segments)} segments from {edf_file_path}")
    return segments, remaining_annotations, edf_duration

def process_events_for_patient(rml_path, edf_group, output_file):
    # Parse annotations once for this patient
    annotations = parse_annotations(rml_path)
    print(f"Processing {len(edf_group)} EDF files for patient with {len(annotations)} events")
    
    all_segments = []
    remaining_annotations = []
    total_elapsed_time = 0  # Tracks cumulative duration of processed EDFs

    for edf_file in edf_group:
        # Adjust event timestamps by subtracting elapsed time
        adjusted_annotations = []
        for event_type, start_time, duration in annotations:
            adjusted_start_time = start_time - total_elapsed_time
            if adjusted_start_time >= 0:
                adjusted_annotations.append((event_type, adjusted_start_time, duration))
            else:
                remaining_annotations.append((event_type, adjusted_start_time + total_elapsed_time, duration))

        # Process EDF with adjusted annotations
        segments, remaining_annotations, edf_duration = preprocess_and_label(edf_file, adjusted_annotations, remaining_annotations)
        all_segments.extend(segments)

        # Update total elapsed time
        total_elapsed_time += edf_duration

    save_to_pkl(all_segments, output_file)
    print(f"Patient data has been saved to {output_file} with {len(all_segments)} segments")

def save_to_pkl(data, output_pkl):
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

# Directory paths and file processing
input_dir = 'data'
output_dir = 'bin'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

all_patient_segments = []  # Collect all segments from all patients

# First, group files by patient
patient_files = {}

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.rml') or file.endswith('.edf'):
            # Extract patient ID from filename (remove [001], [002] etc.)
            patient_id = file.split('[')[0].split('.')[0]
            if patient_id not in patient_files:
                patient_files[patient_id] = {'rml': None, 'edf': []}
            
            if file.endswith('.rml'):
                patient_files[patient_id]['rml'] = os.path.join(root, file)
            elif file.endswith('.edf'):
                patient_files[patient_id]['edf'].append(os.path.join(root, file))

# Process each patient
for patient_id, files in patient_files.items():
    if files['rml'] is None:
        print(f"No RML file found for patient {patient_id}")
        continue
        
    if not files['edf']:
        print(f"No EDF files found for patient {patient_id}")
        continue
        
    print(f"Processing patient: {patient_id}")
    print(f"RML file: {files['rml']}")
    print(f"EDF files: {files['edf']}")
    
    # Sort EDF files to process them in order
    files['edf'].sort()
    
    # Create a temporary output file for this patient
    patient_output = os.path.join(output_dir, f"temp_{patient_id}.pkl")
    
    # Process this patient
    process_events_for_patient(files['rml'], files['edf'], patient_output)
    
    # Load the patient's data and add to the main collection
    with open(patient_output, 'rb') as f:
        patient_segments = pickle.load(f)
        all_patient_segments.extend(patient_segments)
    
    # Remove the temporary file
    os.remove(patient_output)

# Save all patient data to the final output file
final_output_pkl = os.path.join(output_dir, "processed.pkl")
save_to_pkl(all_patient_segments, final_output_pkl)
print(f"All data has been saved to {final_output_pkl} with {len(all_patient_segments)} total segments")