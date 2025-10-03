import os
import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from collections import Counter
import mne

relevant_channels = ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", "EMG Chin", "Leg 1", "Leg 2", "ECG I"]

def parse_annotations(rml_file_path):
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
    events = []
    apnea_count = 0
    normal_count = 0
    
    # DEBUG: Print all event types found
    all_event_types = []
    for event in root.findall('.//ns:Event', namespace):
        event_type = event.get('Type')
        all_event_types.append(event_type)
    
    print(f"DEBUG: All event types in {rml_file_path}: {sorted(set(all_event_types))}")
    
    for event in root.findall('.//ns:Event', namespace):
        event_type = event.get('Type')
        start_time = float(event.get('Start'))
        duration = float(event.get('Duration'))
        
        print(f"DEBUG: Processing event - Type: '{event_type}', Start: {start_time}, Duration: {duration}")
        
        if "apnea" in event_type.lower():
            events.append((event_type, start_time, duration))
            apnea_count += 1
            print(f"DEBUG: Added APNEA event: {event_type}")
        else:
            events.append(("Normal", start_time, duration))
            normal_count += 1
            print(f"DEBUG: Added NORMAL event: {event_type} -> Normal")
    
    print(f"FINAL COUNTS: {apnea_count} apnea events, {normal_count} normal events in {rml_file_path}")
    print(f"Sample events: {events[:5]}")  # Show first 5 events
    
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

    features_df = pd.DataFrame(features)
    print(f"Extracted features from {edf_file_path}, DataFrame shape: {features_df.shape}")
    return features_df, sampling_rates


def preprocess_and_label(edf_file_path, annotations, remaining_annotations, sampling_rate=200):
    """Fixed preprocessing with better event handling"""
    
    print(f"\n📊 PREPROCESSING: {os.path.basename(edf_file_path)}")
    
    # Load EDF
    edf = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    
    # Get duration and calculate segments
    duration_seconds = edf.times[-1]
    segment_length = 30  # seconds
    total_segments = int(duration_seconds // segment_length)
    
    print(f"EDF duration: {duration_seconds:.1f}s, Total possible segments: {total_segments}")
    
    # Separate annotations by type
    apnea_events = []
    normal_events = []
    
    for ann in annotations:
        if 'apnea' in ann['type'].lower():
            apnea_events.append(ann)
        elif ann['type'].lower() == 'normal':
            normal_events.append(ann)
    
    print(f"Input: {len(apnea_events)} apnea events, {len(normal_events)} normal events")
    
    # Create segments for each apnea event
    apnea_segments = []
    processed_apnea = 0
    
    for ann in apnea_events:
        start_time = ann['start']
        end_time = ann['start'] + ann['duration']
        
        # Find which 30s segment this event falls into
        segment_start = int(start_time // segment_length) * segment_length
        
        # Make sure we have enough data for this segment
        if segment_start + segment_length <= duration_seconds:
            try:
                # Extract 30s segment
                start_sample = int(segment_start * sampling_rate)
                end_sample = int((segment_start + segment_length) * sampling_rate)
                
                segment_data = {}
                valid_segment = True
                
                # Extract data for all relevant channels
                for channel in relevant_channels:
                    try:
                        if channel in edf.ch_names:
                            channel_data = edf[channel][0][start_sample:end_sample].flatten()
                            if len(channel_data) > 0:
                                segment_data[channel] = channel_data.tolist()
                            else:
                                valid_segment = False
                                break
                        else:
                            valid_segment = False
                            break
                    except Exception as e:
                        valid_segment = False
                        break
                
                if valid_segment:
                    # Determine the most appropriate label for this segment
                    segment_label = determine_segment_label(ann['type'])
                    
                    segment_data['Label'] = segment_label
                    segment_data['StartTime'] = segment_start
                    segment_data['OriginalEvent'] = ann['type']
                    
                    apnea_segments.append(segment_data)
                    processed_apnea += 1
                
            except Exception as e:
                print(f"Error processing apnea segment at {start_time}: {e}")
                continue
    
    print(f"Successfully created {len(apnea_segments)} apnea segments from {processed_apnea} events")
    
    # Create normal segments (match the number of apnea segments)
    normal_segments = []
    target_normal_count = len(apnea_segments)  # Balance 1:1
    
    if target_normal_count > 0:
        # Create segments from normal periods (avoiding apnea regions)
        apnea_regions = [(ann['start'], ann['start'] + ann['duration']) for ann in apnea_events]
        
        created_normal = 0
        segment_idx = 0
        
        while created_normal < target_normal_count and segment_idx < total_segments:
            segment_start = segment_idx * segment_length
            segment_end = segment_start + segment_length
            
            # Check if this segment overlaps with any apnea event
            overlaps_apnea = False
            for apnea_start, apnea_end in apnea_regions:
                if not (segment_end <= apnea_start or segment_start >= apnea_end):
                    overlaps_apnea = True
                    break
            
            if not overlaps_apnea and segment_end <= duration_seconds:
                try:
                    start_sample = int(segment_start * sampling_rate)
                    end_sample = int(segment_end * sampling_rate)
                    
                    segment_data = {}
                    valid_segment = True
                    
                    for channel in relevant_channels:
                        if channel in edf.ch_names:
                            channel_data = edf[channel][0][start_sample:end_sample].flatten()
                            if len(channel_data) > 0:
                                segment_data[channel] = channel_data.tolist()
                            else:
                                valid_segment = False
                                break
                        else:
                            valid_segment = False
                            break
                    
                    if valid_segment:
                        segment_data['Label'] = 'Normal'
                        segment_data['StartTime'] = segment_start
                        segment_data['OriginalEvent'] = 'Normal'
                        
                        normal_segments.append(segment_data)
                        created_normal += 1
                
                except Exception as e:
                    pass
            
            segment_idx += 1
    
    print(f"Created {len(normal_segments)} normal segments")
    
    # Combine all segments
    all_segments = apnea_segments + normal_segments
    
    print(f"Total segments created: {len(all_segments)}")
    print(f"  Apnea segments: {len(apnea_segments)}")
    print(f"  Normal segments: {len(normal_segments)}")
    
    # Print breakdown by apnea type
    apnea_type_counts = Counter()
    for seg in apnea_segments:
        apnea_type_counts[seg['Label']] += 1
    
    print("Apnea type breakdown:")
    for apnea_type, count in apnea_type_counts.items():
        print(f"  {apnea_type}: {count}")
    
    return all_segments

def determine_segment_label(event_type):
    """Map event type to standardized labels"""
    event_lower = event_type.lower()
    
    if 'obstructive' in event_lower and 'apnea' in event_lower:
        return 'ObstructiveApnea'
    elif 'central' in event_lower and 'apnea' in event_lower:
        return 'CentralApnea'
    elif 'mixed' in event_lower and 'apnea' in event_lower:
        return 'MixedApnea'
    elif 'apnea' in event_lower:
        # Default to obstructive if type is unclear
        return 'ObstructiveApnea'
    else:
        return 'Normal'

def preprocess_and_label_old(edf_file_path, annotations, remaining_annotations, sampling_rate=200):
    features_df, sampling_rates = extract_features_from_edf(edf_file_path)
    
    segments = []
    window_size = 60 * sampling_rate  # 60 seconds
    edf_duration = len(features_df) / sampling_rate
    print(f"EDF duration: {edf_duration} seconds, Sampling rate: {sampling_rate}, DataFrame length: {len(features_df)}")

    # DEBUG: Print all incoming annotations
    print(f"DEBUG: Total annotations received: {len(annotations)}")
    for i, (event_type, start_time, duration) in enumerate(annotations[:10]):  # First 10
        print(f"  Annotation {i}: Type='{event_type}', Start={start_time}, Duration={duration}")

    # Separate apnea and normal events that will be processed in this EDF
    apnea_events = [event for event in annotations if "apnea" in event[0].lower() and event[1] < edf_duration]
    all_normal_events = [event for event in annotations if "apnea" not in event[0].lower() and event[1] < edf_duration]
    
    apnea_count = len(apnea_events)
    total_normal_count = len(all_normal_events)
    
    print(f"DEBUG: Found {apnea_count} apnea events and {total_normal_count} normal events in this EDF")
    
    if apnea_count > 0:
        print("DEBUG: Sample apnea events:")
        for i, event in enumerate(apnea_events[:3]):
            print(f"  Apnea {i}: {event}")
    else:
        print("DEBUG: NO APNEA EVENTS FOUND!")
        
    if total_normal_count > 0:
        print("DEBUG: Sample normal events:")
        for i, event in enumerate(all_normal_events[:3]):
            print(f"  Normal {i}: {event}")
    
    # Check if we have any apnea events
    if apnea_count == 0:
        print("WARNING: No apnea events found in this EDF! All segments will be Normal.")
        all_events = all_normal_events  # Use all normal events
    elif total_normal_count > apnea_count:
        # FIXED: Randomly sample normal events to match apnea count
        # Convert to numpy array of indices, then select by indices
        normal_indices = np.random.choice(len(all_normal_events), size=apnea_count, replace=False)
        normal_events = [all_normal_events[i] for i in normal_indices]
        print(f"Downsampled {total_normal_count} normal events to {len(normal_events)} to match {apnea_count} apnea events")
        all_events = apnea_events + normal_events
    else:
        normal_events = all_normal_events
        print(f"Using all {len(normal_events)} normal events with {apnea_count} apnea events")
        all_events = apnea_events + normal_events

    print(f"DEBUG: Final event counts for processing - Apnea: {len([e for e in all_events if 'apnea' in e[0].lower()])}, Normal: {len([e for e in all_events if 'apnea' not in e[0].lower()])}")

    # Process events
    processed_segments = 0
    skipped_segments = 0
    
    for event in all_events:
        label, start_time, _ = event

        if start_time >= edf_duration:
            remaining_annotations.append((label, start_time - edf_duration, _))
            skipped_segments += 1
            continue

        start_idx = int(start_time * sampling_rate)
        end_idx = start_idx + window_size

        # Ensure we don't go beyond the data
        if end_idx > len(features_df):
            skipped_segments += 1
            continue

        segment_data = {}
        for channel in features_df.columns:
            if channel in sampling_rates and sampling_rates[channel] is not None:
                channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                # Pad with zeros if needed
                if len(channel_data) < window_size:
                    channel_data += [0.0] * (window_size - len(channel_data))
                segment_data[channel] = channel_data
            else:
                segment_data[channel] = [0.0] * window_size

        segment_data['Label'] = label
        segments.append(segment_data)
        processed_segments += 1

    print(f"DEBUG: Processed {processed_segments} segments, skipped {skipped_segments} segments")
    
    # Count final labels
    final_labels = [seg['Label'] for seg in segments]
    label_counts = {}
    for label in final_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"DEBUG: Final segment label counts: {label_counts}")

    print(f"Generated {len(segments)} segments from {edf_file_path}")
    return segments, remaining_annotations, edf_duration

def process_events_for_patient(rml_path, edf_group, output_file):
    try:
        annotations = parse_annotations(rml_path)
        print(f"Processing {len(edf_group)} EDF files for patient with {len(annotations)} events")

        all_segments = []
        total_elapsed_time = 0

        for edf_file in edf_group:
            try:
                # Adjust event timestamps by subtracting elapsed time
                adjusted_annotations = []
                for event_type, start_time, duration in annotations:
                    adjusted_start_time = start_time - total_elapsed_time
                    if adjusted_start_time >= 0:
                        adjusted_annotations.append((event_type, adjusted_start_time, duration))
                
                # Process this EDF
                segments, remaining_annotations, edf_duration = preprocess_and_label(edf_file, adjusted_annotations, [])
                all_segments.extend(segments)
                
                # Update annotations for next EDF
                annotations = []
                for event_type, start_time, duration in adjusted_annotations:
                    if start_time >= edf_duration:
                        annotations.append((event_type, start_time - edf_duration + total_elapsed_time + edf_duration, duration))
                
                total_elapsed_time += edf_duration
                
            except Exception as e:
                print(f"Error processing EDF {edf_file}: {e}")
                continue

        save_to_pkl(all_segments, output_file)
        print(f"Patient data has been saved to {output_file} with {len(all_segments)} segments")
        
    except Exception as e:
        print(f"Error processing patient {rml_path}: {e}")
        # Create empty file to avoid errors
        save_to_pkl([], output_file)

def save_to_pkl(data, output_pkl):
    with open(output_pkl, 'wb') as f:
        pickle.dump(data, f)

# Directory paths and file processing
input_dir = '/raid/userdata/cybulsm/ThesisProj/dataset'
output_dir = '/raid/userdata/cybulskm/ThesisProj/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

all_patient_segments = []

# FIXED: Single file grouping loop
patient_files = {}

for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.rml') or file.endswith('.edf'):
            print(f"Processing file: {file}")

            # Extract patient ID from filename
            if '[' in file:
                patient_id = file.split('[')[0]
            else:
                patient_id = file.split('.')[0].split('_')[0]

            print(f"Extracted patient ID: {patient_id}")

            if patient_id not in patient_files:
                patient_files[patient_id] = {'rml': None, 'edf': []}

            if file.endswith('.rml'):
                patient_files[patient_id]['rml'] = os.path.join(root, file)
            elif file.endswith('.edf'):
                patient_files[patient_id]['edf'].append(os.path.join(root, file))

# Debug output
print(f"Found {len(patient_files)} patients:")
for patient_id, files in patient_files.items():
    print(f"Patient {patient_id}: RML={files['rml'] is not None}, EDF count={len(files['edf'])}")

# Process each patient with error handling
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

    try:
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

    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        print(f"Skipping patient {patient_id}")
        # Clean up temporary file if it exists
        patient_output = os.path.join(output_dir, f"temp_{patient_id}.pkl")
        if os.path.exists(patient_output):
            os.remove(patient_output)
        continue

# Save all patient data to the final output file
final_output_pkl = os.path.join(output_dir, "processed.pkl")
save_to_pkl(all_patient_segments, final_output_pkl)
print(f"All data has been saved to {final_output_pkl} with {len(all_patient_segments)} total segments")

# Add this diagnostic function first:

def diagnose_preprocessing_issue(edf_file_path, rml_file_path):
    """Diagnose what's happening during preprocessing"""
    print(f"\n🔍 DIAGNOSING: {os.path.basename(edf_file_path)}")
    
    # 1. Check annotations
    annotations = parse_annotations(rml_file_path)
    print(f"Raw annotations found: {len(annotations)}")
    
    apnea_count = 0
    normal_count = 0
    for ann in annotations:
        if 'apnea' in ann['type'].lower():
            apnea_count += 1
        elif ann['type'].lower() == 'normal':
            normal_count += 1
    
    print(f"Apnea annotations: {apnea_count}")
    print(f"Normal annotations: {normal_count}")
    
    # 2. Check EDF file
    try:
        edf = mne.io.read_raw_edf(edf_file_path, preload=False, verbose=False)
        duration_seconds = edf.times[-1]
        print(f"EDF duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.1f} hours)")
        
        # 3. Check segment creation
        segment_length = 30  # seconds
        total_possible_segments = int(duration_seconds // segment_length)
        print(f"Possible 30s segments: {total_possible_segments}")
        
    except Exception as e:
        print(f"Error reading EDF: {e}")

# Update your main preprocessing function:
def preprocess_and_label(edf_file_path, annotations, remaining_annotations, sampling_rate=200):
    """Fixed preprocessing with better event handling"""
    
    print(f"\n📊 PREPROCESSING: {os.path.basename(edf_file_path)}")
    
    # Load EDF
    edf = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
    
    # Get duration and calculate segments
    duration_seconds = edf.times[-1]
    segment_length = 30  # seconds
    total_segments = int(duration_seconds // segment_length)
    
    print(f"EDF duration: {duration_seconds:.1f}s, Total possible segments: {total_segments}")
    
    # Separate annotations by type
    apnea_events = []
    normal_events = []
    
    for ann in annotations:
        if 'apnea' in ann['type'].lower():
            apnea_events.append(ann)
        elif ann['type'].lower() == 'normal':
            normal_events.append(ann)
    
    print(f"Input: {len(apnea_events)} apnea events, {len(normal_events)} normal events")
    
    # Create segments for each apnea event
    apnea_segments = []
    processed_apnea = 0
    
    for ann in apnea_events:
        start_time = ann['start']
        end_time = ann['start'] + ann['duration']
        
        # Find which 30s segment this event falls into
        segment_start = int(start_time // segment_length) * segment_length
        
        # Make sure we have enough data for this segment
        if segment_start + segment_length <= duration_seconds:
            try:
                # Extract 30s segment
                start_sample = int(segment_start * sampling_rate)
                end_sample = int((segment_start + segment_length) * sampling_rate)
                
                segment_data = {}
                valid_segment = True
                
                # Extract data for all relevant channels
                for channel in relevant_channels:
                    try:
                        if channel in edf.ch_names:
                            channel_data = edf[channel][0][start_sample:end_sample].flatten()
                            if len(channel_data) > 0:
                                segment_data[channel] = channel_data.tolist()
                            else:
                                valid_segment = False
                                break
                        else:
                            valid_segment = False
                            break
                    except Exception as e:
                        valid_segment = False
                        break
                
                if valid_segment:
                    # Determine the most appropriate label for this segment
                    segment_label = determine_segment_label(ann['type'])
                    
                    segment_data['Label'] = segment_label
                    segment_data['StartTime'] = segment_start
                    segment_data['OriginalEvent'] = ann['type']
                    
                    apnea_segments.append(segment_data)
                    processed_apnea += 1
                
            except Exception as e:
                print(f"Error processing apnea segment at {start_time}: {e}")
                continue
    
    print(f"Successfully created {len(apnea_segments)} apnea segments from {processed_apnea} events")
    
    # Create normal segments (match the number of apnea segments)
    normal_segments = []
    target_normal_count = len(apnea_segments)  # Balance 1:1
    
    if target_normal_count > 0:
        # Create segments from normal periods (avoiding apnea regions)
        apnea_regions = [(ann['start'], ann['start'] + ann['duration']) for ann in apnea_events]
        
        created_normal = 0
        segment_idx = 0
        
        while created_normal < target_normal_count and segment_idx < total_segments:
            segment_start = segment_idx * segment_length
            segment_end = segment_start + segment_length
            
            # Check if this segment overlaps with any apnea event
            overlaps_apnea = False
            for apnea_start, apnea_end in apnea_regions:
                if not (segment_end <= apnea_start or segment_start >= apnea_end):
                    overlaps_apnea = True
                    break
            
            if not overlaps_apnea and segment_end <= duration_seconds:
                try:
                    start_sample = int(segment_start * sampling_rate)
                    end_sample = int(segment_end * sampling_rate)
                    
                    segment_data = {}
                    valid_segment = True
                    
                    for channel in relevant_channels:
                        if channel in edf.ch_names:
                            channel_data = edf[channel][0][start_sample:end_sample].flatten()
                            if len(channel_data) > 0:
                                segment_data[channel] = channel_data.tolist()
                            else:
                                valid_segment = False
                                break
                        else:
                            valid_segment = False
                            break
                    
                    if valid_segment:
                        segment_data['Label'] = 'Normal'
                        segment_data['StartTime'] = segment_start
                        segment_data['OriginalEvent'] = 'Normal'
                        
                        normal_segments.append(segment_data)
                        created_normal += 1
                
                except Exception as e:
                    pass
            
            segment_idx += 1
    
    print(f"Created {len(normal_segments)} normal segments")
    
    # Combine all segments
    all_segments = apnea_segments + normal_segments
    
    print(f"Total segments created: {len(all_segments)}")
    print(f"  Apnea segments: {len(apnea_segments)}")
    print(f"  Normal segments: {len(normal_segments)}")
    
    # Print breakdown by apnea type
    apnea_type_counts = Counter()
    for seg in apnea_segments:
        apnea_type_counts[seg['Label']] += 1
    
    print("Apnea type breakdown:")
    for apnea_type, count in apnea_type_counts.items():
        print(f"  {apnea_type}: {count}")
    
    return all_segments

def main():
    # ... existing code ...
    
    # Add diagnostic check for first few files
    for i, (edf_file, rml_file) in enumerate(zip(edf_files[:3], rml_files[:3])):
        print(f"\n{'='*60}")
        diagnose_preprocessing_issue(edf_file, rml_file)
        
        # Then run normal preprocessing
        annotations = parse_annotations(rml_file)
        segments = preprocess_and_label(edf_file, annotations, [], sampling_rate=200)
        
        if i == 0:  # Just process first file for debugging
            break