import os
import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from collections import Counter

relevant_channels = ["EEG A1-A2", "EEG C3-A2", "EEG C4-A1", "EOG LOC-A2", "EOG ROC-A2", "EMG Chin", "Leg 1", "Leg 2", "ECG I"]

def parse_annotations(rml_file_path):
    """Parse RML file and extract all events with proper apnea classification"""
    tree = ET.parse(rml_file_path)
    root = tree.getroot()
    namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
    events = []
    
    print(f"\n🔍 Parsing annotations from: {os.path.basename(rml_file_path)}")
    
    apnea_count = 0
    normal_count = 0
    event_types = Counter()
    
    for event in root.findall('.//ns:Event', namespace):
        event_type = event.get('Type')
        start_time = float(event.get('Start'))
        duration = float(event.get('Duration'))
        
        event_types[event_type] += 1
        
        # Classify events properly
        if "apnea" in event_type.lower():
            events.append(("Apnea", start_time, duration))
            apnea_count += 1
        else:
            # All non-apnea events are normal
            events.append(("Normal", start_time, duration))
            normal_count += 1
    
    print(f"📊 Event summary:")
    print(f"  Total events: {len(events)}")
    print(f"  Apnea events: {apnea_count}")
    print(f"  Normal events: {normal_count}")
    print(f"  Event types found: {dict(event_types)}")
    
    return events

def extract_features_from_edf(edf_file_path):
    """Extract signal data from EDF file"""
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
                print(f"⚠️ Channel {channel} not found in the EDF file.")
                features[channel] = None
                sampling_rates[channel] = None

    edf_file.close()
    features_df = pd.DataFrame(features)
    return features_df, sampling_rates

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

def preprocess_and_label(edf_file_path, annotations, sampling_rate=200):
    """Fixed preprocessing that creates segments for ALL apnea events"""
    
    print(f"\n📊 Processing EDF: {os.path.basename(edf_file_path)}")
    
    # Load EDF data
    features_df, sampling_rates = extract_features_from_edf(edf_file_path)
    edf_duration = len(features_df) / sampling_rate
    
    print(f"EDF duration: {edf_duration:.1f}s, DataFrame shape: {features_df.shape}")
    
    # Separate events by type
    apnea_events = []
    normal_events = []
    
    for event_type, start_time, duration in annotations:
        if start_time < edf_duration:  # Only events within this EDF
            if "apnea" in event_type.lower():
                apnea_events.append((event_type, start_time, duration))
            else:
                normal_events.append((event_type, start_time, duration))
    
    print(f"Events in this EDF: {len(apnea_events)} apnea, {len(normal_events)} normal")
    
    segments = []
    segment_length = 30  # seconds
    window_size = segment_length * sampling_rate
    
    # Process ALL apnea events - create segments based on event timing
    processed_apnea = 0
    apnea_type_counts = Counter()
    
    for event_type, start_time, duration in apnea_events:
        # Create a 30s segment that contains this apnea event
        # Center the segment around the apnea event
        event_center = start_time + (duration / 2)
        segment_start = max(0, event_center - (segment_length / 2))
        
        # Ensure segment doesn't go beyond EDF duration
        if segment_start + segment_length > edf_duration:
            segment_start = edf_duration - segment_length
        
        if segment_start >= 0:
            start_idx = int(segment_start * sampling_rate)
            end_idx = start_idx + window_size
            
            # Ensure we don't exceed data bounds
            if end_idx <= len(features_df):
                segment_data = {}
                valid_segment = True
                
                # Extract data for all channels
                for channel in relevant_channels:
                    if channel in features_df.columns and sampling_rates.get(channel) is not None:
                        channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                        
                        # Ensure exact window size
                        if len(channel_data) < window_size:
                            channel_data += [0.0] * (window_size - len(channel_data))
                        elif len(channel_data) > window_size:
                            channel_data = channel_data[:window_size]
                        
                        segment_data[channel] = channel_data
                    else:
                        # Fill missing channels with zeros
                        segment_data[channel] = [0.0] * window_size
                
                # Determine proper label
                label = determine_segment_label(event_type)
                segment_data['Label'] = label
                segment_data['StartTime'] = segment_start
                segment_data['OriginalEvent'] = event_type
                segment_data['EventStart'] = start_time
                segment_data['EventDuration'] = duration
                
                segments.append(segment_data)
                processed_apnea += 1
                apnea_type_counts[label] += 1
    
    print(f"✅ Created {processed_apnea} apnea segments")
    for apnea_type, count in apnea_type_counts.items():
        print(f"  {apnea_type}: {count}")
    
    # Create normal segments to balance the dataset
    # Target: equal number of normal segments as total apnea segments
    target_normal_count = len(segments)
    
    if target_normal_count > 0 and normal_events:
        print(f"🎯 Creating {target_normal_count} normal segments...")
        
        # Get time periods occupied by apnea events (with some buffer)
        occupied_periods = []
        for event_type, start_time, duration in apnea_events:
            # Add 15s buffer on each side
            buffer_start = max(0, start_time - 15)
            buffer_end = min(edf_duration, start_time + duration + 15)
            occupied_periods.append((buffer_start, buffer_end))
        
        # Create normal segments from non-apnea periods
        normal_segments_created = 0
        max_attempts = target_normal_count * 3  # Prevent infinite loops
        attempts = 0
        
        while normal_segments_created < target_normal_count and attempts < max_attempts:
            attempts += 1
            
            # Pick a random time that allows for a full 30s segment
            random_start = np.random.uniform(0, max(0, edf_duration - segment_length))
            segment_end = random_start + segment_length
            
            # Check if this overlaps with any apnea period
            overlaps = False
            for occupied_start, occupied_end in occupied_periods:
                if not (segment_end <= occupied_start or random_start >= occupied_end):
                    overlaps = True
                    break
            
            if not overlaps:
                start_idx = int(random_start * sampling_rate)
                end_idx = start_idx + window_size
                
                if end_idx <= len(features_df):
                    segment_data = {}
                    
                    for channel in relevant_channels:
                        if channel in features_df.columns and sampling_rates.get(channel) is not None:
                            channel_data = features_df[channel].iloc[start_idx:end_idx].tolist()
                            
                            if len(channel_data) < window_size:
                                channel_data += [0.0] * (window_size - len(channel_data))
                            elif len(channel_data) > window_size:
                                channel_data = channel_data[:window_size]
                        
                            segment_data[channel] = channel_data
                        else:
                            segment_data[channel] = [0.0] * window_size
                    
                    segment_data['Label'] = 'Normal'
                    segment_data['StartTime'] = random_start
                    segment_data['OriginalEvent'] = 'Normal'
                    
                    segments.append(segment_data)
                    normal_segments_created += 1
        
        print(f"✅ Created {normal_segments_created} normal segments")
    
    # Final segment count
    final_labels = Counter(seg['Label'] for seg in segments)
    print(f"📊 Final segments: {dict(final_labels)} (Total: {len(segments)})")
    
    return segments

def process_patient(patient_id, rml_path, edf_files):
    """Process all files for a single patient"""
    print(f"\n{'='*60}")
    print(f"🏥 Processing Patient: {patient_id}")
    print(f"📁 RML file: {os.path.basename(rml_path)}")
    print(f"📁 EDF files ({len(edf_files)}): {[os.path.basename(f) for f in edf_files]}")
    
    try:
        # Parse annotations from RML file
        annotations = parse_annotations(rml_path)
        
        if not annotations:
            print(f"❌ No annotations found for patient {patient_id}")
            return []
        
        all_segments = []
        
        # Process each EDF file for this patient
        for i, edf_file in enumerate(sorted(edf_files)):
            print(f"\n📄 Processing EDF {i+1}/{len(edf_files)}: {os.path.basename(edf_file)}")
            
            try:
                segments = preprocess_and_label(edf_file, annotations)
                all_segments.extend(segments)
                print(f"✅ Added {len(segments)} segments from this EDF")
                
            except Exception as e:
                print(f"❌ Error processing EDF {edf_file}: {e}")
                continue
        
        print(f"\n🎯 Patient {patient_id} Summary:")
        print(f"  Total segments: {len(all_segments)}")
        
        if all_segments:
            patient_labels = Counter(seg['Label'] for seg in all_segments)
            for label, count in patient_labels.items():
                print(f"  {label}: {count}")
        
        return all_segments
        
    except Exception as e:
        print(f"❌ Error processing patient {patient_id}: {e}")
        return []

def main():
    """Main processing function"""
    print("🚀 STARTING 285-PATIENT PREPROCESSING")
    print("=" * 60)
    
    # Directory paths
    input_dir = '/raid/userdata/cybulskm/ThesisProj/dataset'
    output_dir = '/raid/userdata/cybulskm/ThesisProj/'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all RML files
    print("🔍 Scanning for RML files...")
    rml_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.rml'):
                rml_files.append(os.path.join(root, file))
    
    print(f"Found {len(rml_files)} RML files total")
    
    # Sort and limit to 285 patients
    rml_files.sort()
    MAX_PATIENTS = 285
    
    if len(rml_files) > MAX_PATIENTS:
        print(f"📊 Limiting to first {MAX_PATIENTS} patients")
        rml_files = rml_files[:MAX_PATIENTS]
    else:
        print(f"📊 Processing all {len(rml_files)} available patients")
    
    # Build patient-file mapping
    print("\n🗂️ Building patient-file mapping...")
    patient_files = {}
    
    for rml_path in rml_files:
        rml_file = os.path.basename(rml_path)
        
        # Extract patient ID
        if '[' in rml_file:
            patient_id = rml_file.split('[')[0]
        else:
            patient_id = rml_file.split('.')[0].split('_')[0]
        
        # Find corresponding EDF files
        patient_dir = os.path.dirname(rml_path)
        edf_files = []
        
        for root, _, files in os.walk(patient_dir):
            for file in files:
                if file.endswith('.edf'):
                    # Check if EDF belongs to this patient
                    if '[' in file:
                        edf_patient_id = file.split('[')[0]
                    else:
                        edf_patient_id = file.split('.')[0].split('_')[0]
                    
                    if edf_patient_id == patient_id:
                        edf_files.append(os.path.join(root, file))
        
        if edf_files:  # Only include patients with EDF files
            patient_files[patient_id] = {
                'rml': rml_path,
                'edf': edf_files
            }
    
    print(f"✅ Found {len(patient_files)} patients with both RML and EDF files")
    
    # Process each patient
    all_segments = []
    processed_count = 0
    skipped_count = 0
    
    for i, (patient_id, files) in enumerate(patient_files.items(), 1):
        print(f"\n🔄 Processing patient {i}/{len(patient_files)}")
        
        try:
            patient_segments = process_patient(patient_id, files['rml'], files['edf'])
            
            if patient_segments:
                all_segments.extend(patient_segments)
                processed_count += 1
                print(f"✅ Successfully processed patient {patient_id}: {len(patient_segments)} segments")
            else:
                skipped_count += 1
                print(f"❌ No segments generated for patient {patient_id}")
                
        except Exception as e:
            print(f"❌ Error processing patient {patient_id}: {e}")
            skipped_count += 1
            continue
    
    # Final summary
    print(f"\n🎯 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Patients successfully processed: {processed_count}")
    print(f"Patients skipped: {skipped_count}")
    print(f"Total segments generated: {len(all_segments):,}")
    
    if all_segments:
        # Final label distribution
        final_labels = Counter(seg['Label'] for seg in all_segments)
        print(f"\n📊 Final Label Distribution:")
        for label, count in final_labels.items():
            percentage = (count / len(all_segments)) * 285
            print(f"  {label}: {count:,} segments ({percentage:.1f}%)")
        
        # Check balance
        apnea_total = sum(count for label, count in final_labels.items() if label != 'Normal')
        normal_total = final_labels.get('Normal', 0)
        
        if apnea_total > 0:
            balance_ratio = normal_total / apnea_total
            print(f"\n⚖️ Balance Analysis:")
            print(f"  Normal segments: {normal_total:,}")
            print(f"  Total apnea segments: {apnea_total:,}")
            print(f"  Normal/Apnea ratio: {balance_ratio:.2f}:1")
        
        # Save to file
        output_file = os.path.join(output_dir, "285_patients_processed.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(all_segments, f)
        
        print(f"\n💾 Data saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")
        
    else:
        print("❌ No segments were generated!")

if __name__ == "__main__":
    main()