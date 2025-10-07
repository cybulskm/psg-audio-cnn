import os
import pandas as pd
import numpy as np
import pyedflib
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from collections import Counter
from scipy import signal
import statistics

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
    signals = {}
    sampling_rates = {}

    for channel in signal_labels:
        try:
            signal_index = signal_labels.index(channel)
            signals[channel] = edf_file.readSignal(signal_index)
            sampling_rates[channel] = edf_file.getSampleFrequency(signal_index)
        except ValueError:
            print(f"⚠️ Channel {channel} not found in the EDF file.")
            signals[channel] = None
            sampling_rates[channel] = None

    edf_file.close()
    return signals, sampling_rates

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
    elif "hypopnea" in event_lower:
        return 'Hypopnea'
    else:
        return 'Normal'

def get_edf_duration(edf_file_path):
    """Get duration of EDF file in seconds"""
    edf_file = pyedflib.EdfReader(edf_file_path)
    duration = edf_file.getFileDuration()
    edf_file.close()
    return duration

def find_most_common_frequency(edf_files):
    """Determine the most common sampling frequency across all files and channels"""
    all_frequencies = []
    
    for edf_file in edf_files:
        with pyedflib.EdfReader(edf_file) as edf:
            n_signals = edf.signals_in_file
            for i in range(n_signals):
                all_frequencies.append(edf.getSampleFrequency(i))
    
    most_common_freq = statistics.mode(all_frequencies)
    print(f"\n📊 Most common sampling frequency: {most_common_freq} Hz")
    return most_common_freq

def resample_signal(signal_data, orig_freq, target_freq):
    """Resample signal to target frequency"""
    if orig_freq == target_freq:
        return signal_data
    
    # Calculate number of samples for target frequency
    n_samples = int(len(signal_data) * target_freq / orig_freq)
    return signal.resample(signal_data, n_samples)

def process_edf_file(edf_file_path, file_events, file_start_time, target_freq):
    """Process all events for a single EDF file at once"""
    segments = []
    window_size = 60  # Fixed 60-second window
    
    try:
        with pyedflib.EdfReader(edf_file_path) as edf_file:
            duration = edf_file.getFileDuration()
            signal_labels = edf_file.getSignalLabels()
            
            # Load all signals at once and resample to target frequency
            signals = {}
            for channel in signal_labels:
                try:
                    signal_index = signal_labels.index(channel)
                    orig_freq = edf_file.getSampleFrequency(signal_index)
                    signal_data = edf_file.readSignal(signal_index)
                    
                    # Resample to target frequency
                    resampled_signal = resample_signal(signal_data, orig_freq, target_freq)
                    signals[channel] = resampled_signal
                except ValueError:
                    print(f"⚠️ Channel {channel} not found")
                    signals[channel] = None
            
            # Process all events for this file
            for event_type, start_time, event_duration in file_events:
                segment_data = {}
                
                # Convert global time to local file time
                local_start = start_time - file_start_time
                
                # Calculate window boundaries centered on event
                event_center = local_start + (event_duration / 2)
                window_start = event_center - (window_size / 2)
                window_end = event_center + (window_size / 2)
                
                # Ensure window is within file bounds
                if 0 <= window_start < duration and window_end <= duration:
                    for channel, signal in signals.items():
                        if signal is not None:
                            # Convert time to samples using target frequency
                            start_sample = int(window_start * target_freq)
                            end_sample = int(window_end * target_freq)
                            
                            # Extract exactly 60 seconds worth of samples
                            samples_needed = int(window_size * target_freq)
                            
                            if start_sample >= 0 and end_sample <= len(signal):
                                try:
                                    window_data = signal[start_sample:end_sample]
                                    if len(window_data) == samples_needed:
                                        segment_data[channel] = window_data.tolist()
                                    else:
                                        print(f"⚠️ Window size mismatch for {channel}")
                                except Exception as e:
                                    print(f"⚠️ Error with {channel}: {e}")
                    
                    if segment_data:
                        segment_data.update({
                            'Label': determine_segment_label(event_type),
                            'EventType': event_type,
                            'Duration': event_duration,
                            'StartTime': start_time,
                            'FileStart': file_start_time,
                            'WindowStart': window_start + file_start_time,  # Convert back to global time
                            'WindowEnd': window_end + file_start_time,      # Convert back to global time
                            'SamplingRate': target_freq
                        })
                        segments.append(segment_data)
            
            # Explicitly delete large objects
            del signals
            
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(edf_file_path)}: {e}")
    
    return segments

def preprocess_and_label(edf_files, annotations):
    """Process events across multiple EDF files"""
    print("\n📊 Processing EDF files...")
    
    # Find most common sampling frequency
    target_freq = find_most_common_frequency(edf_files)
    
    # Map events to files
    file_durations = []
    cumulative_duration = 0
    file_events = {}
    
    # Calculate file durations first
    for edf_file in sorted(edf_files):
        duration = get_edf_duration(edf_file)
        file_info = {
            'file': edf_file,
            'start': cumulative_duration,
            'end': cumulative_duration + duration,
            'duration': duration
        }
        file_durations.append(file_info)
        file_events[edf_file] = []
        cumulative_duration += duration
    
    # Map events to files
    for event in annotations:
        event_type, start_time, duration = event
        event_end = start_time + duration
        
        for file_info in file_durations:
            if not (event_end <= file_info['start'] or start_time >= file_info['end']):
                file_events[file_info['file']].append(event)
    
    # Process each file once
    all_segments = []
    for file_info in file_durations:
        if file_events[file_info['file']]:
            segments = process_edf_file(
                file_info['file'],
                file_events[file_info['file']],
                file_info['start'],
                target_freq
            )
            all_segments.extend(segments)
    
    return all_segments

def save_results(segments, output_file):
    """Save results as CSV with one row per event"""
    if not segments:
        return
    
    # Get all column names
    columns = ['Label', 'EventType'] + [col for col in segments[0].keys() 
                                      if col not in ['Label', 'EventType']]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(segments)

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

        # Process all EDF files together
        segments = preprocess_and_label(edf_files, annotations)
        
        print(f"\n🎯 Patient {patient_id} Summary:")
        print(f"  Total segments: {len(segments)}")

        if segments:
            patient_labels = Counter(seg['Label'] for seg in segments)
            for label, count in patient_labels.items():
                print(f"  {label}: {count}")

        return segments

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
        output_file = os.path.join(output_dir, "285_patients_processed_v2.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(all_segments, f)
        
        print(f"\n💾 Data saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1e6:.1f} MB")
        
    else:
        print("❌ No segments were generated!")

if __name__ == "__main__":
    main()