import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import pandas as pd
from datetime import datetime

def parse_rml_file(rml_path):
    """Parse a single RML file and extract apnea events using correct XML structure"""
    apnea_events = []
    
    try:
        tree = ET.parse(rml_path)
        root = tree.getroot()
        
        # Use the correct namespace from preprocess.py
        namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
        
        # Find all Event elements (not ScoredEvent)
        events_found = 0
        apnea_events_found = 0
        
        for event in root.findall('.//ns:Event', namespace):
            events_found += 1
            event_type = event.get('Type')
            start_time = event.get('Start')
            duration = event.get('Duration')
            
            if event_type and start_time and duration:
                try:
                    start_time_float = float(start_time)
                    duration_float = float(duration)
                    
                    # Only include events with "apnea" in the name (case insensitive)
                    if 'apnea' in event_type.lower():
                        apnea_events.append({
                            'name': event_type,
                            'start': start_time_float,
                            'duration': duration_float,
                            'end': start_time_float + duration_float
                        })
                        apnea_events_found += 1
                except (ValueError, TypeError):
                    continue
        
        if events_found > 0:
            print(f"  DEBUG: Found {events_found} total events, {apnea_events_found} apnea events")
        
        return apnea_events
        
    except Exception as e:
        print(f"Error parsing {rml_path}: {e}")
        return []

def explore_all_rml_files(base_directory):
    """Explore all RML files in the directory structure"""
    print("🔍 EXPLORING ALL RML FILES FOR APNEA EVENTS")
    print("=" * 60)
    
    rml_files = []
    apnea_stats = []
    all_event_names = Counter()
    patient_stats = {}
    
    # Find all RML files
    print("Finding RML files...")
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.rml'):
                rml_path = os.path.join(root, file)
                rml_files.append(rml_path)
    
    print(f"Found {len(rml_files)} RML files")
    print()
    
    # Process each RML file
    total_apnea_events = 0
    files_with_apnea = 0
    files_without_apnea = 0
    
    for i, rml_path in enumerate(rml_files):
        print(f"Processing {i+1}/{len(rml_files)}: {os.path.basename(rml_path)}")
        
        # Extract patient ID from filename (same logic as preprocess.py)
        filename = os.path.basename(rml_path)
        if '[' in filename:
            patient_id = filename.split('[')[0]
        else:
            patient_id = filename.split('.')[0].split('_')[0]
        
        # Parse the file
        apnea_events = parse_rml_file(rml_path)
        
        # Count event types
        event_type_counts = Counter()
        for event in apnea_events:
            event_type_counts[event['name']] += 1
            all_event_names[event['name']] += 1
        
        # Calculate total duration
        total_duration = sum(event['duration'] for event in apnea_events)
        
        # Store statistics
        file_stats = {
            'file_path': rml_path,
            'patient_id': patient_id,
            'total_apnea_events': len(apnea_events),
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'event_types': dict(event_type_counts),
            'unique_event_types': len(event_type_counts)
        }
        
        apnea_stats.append(file_stats)
        
        # Update patient statistics
        if patient_id not in patient_stats:
            patient_stats[patient_id] = {
                'files': 0,
                'total_events': 0,
                'total_duration': 0,
                'event_types': Counter()
            }
        
        patient_stats[patient_id]['files'] += 1
        patient_stats[patient_id]['total_events'] += len(apnea_events)
        patient_stats[patient_id]['total_duration'] += total_duration
        patient_stats[patient_id]['event_types'].update(event_type_counts)
        
        # Track files with/without apnea
        if len(apnea_events) > 0:
            files_with_apnea += 1
        else:
            files_without_apnea += 1
        
        total_apnea_events += len(apnea_events)
        
        # Print summary for this file
        if len(apnea_events) > 0:
            print(f"  ✅ Found {len(apnea_events)} apnea events ({total_duration/60:.1f} min total)")
            for event_type, count in event_type_counts.most_common():
                print(f"     - {event_type}: {count}")
        else:
            print(f"  ❌ No apnea events found")
        print()
        
        # Stop early for debugging if we find some events
        if i < 10 and len(apnea_events) > 0:
            print(f"  DEBUG: Sample events from this file:")
            for j, event in enumerate(apnea_events[:3]):
                print(f"    Event {j}: {event['name']} at {event['start']:.1f}s for {event['duration']:.1f}s")
    
    # Print overall summary
    print("=" * 60)
    print("📊 OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total RML files processed: {len(rml_files)}")
    print(f"Files with apnea events: {files_with_apnea}")
    print(f"Files without apnea events: {files_without_apnea}")
    print(f"Total apnea events found: {total_apnea_events}")
    print(f"Average events per file: {total_apnea_events/len(rml_files):.1f}")
    print()
    
    # Print all unique apnea event types found
    if total_apnea_events > 0:
        print("🏷️ ALL APNEA EVENT TYPES FOUND:")
        print("-" * 40)
        for event_type, count in all_event_names.most_common():
            percentage = (count / total_apnea_events) * 100
            print(f"{event_type:30s}: {count:6d} ({percentage:5.1f}%)")
        print()
    else:
        print("🏷️ NO APNEA EVENT TYPES FOUND")
        print("-" * 40)
        print("This suggests there might be an issue with:")
        print("• XML parsing (wrong namespace/structure)")
        print("• Event naming (events might not contain 'apnea')")
        print("• File format or corruption")
        print()
        
        # Let's debug the first file
        if rml_files:
            print("🔍 DEBUGGING FIRST RML FILE:")
            print("-" * 40)
            debug_rml_structure(rml_files[0])
    
    # Print patient-level summary
    print("👥 PATIENT-LEVEL SUMMARY:")
    print("-" * 60)
    print(f"{'Patient ID':15s} {'Files':>6s} {'Events':>8s} {'Duration(min)':>12s} {'Event Types':>12s}")
    print("-" * 60)
    
    for patient_id, stats in sorted(patient_stats.items()):
        duration_min = stats['total_duration'] / 60
        event_types = len(stats['event_types'])
        print(f"{patient_id:15s} {stats['files']:6d} {stats['total_events']:8d} {duration_min:12.1f} {event_types:12d}")
    
    # Find patients with most/least apnea events
    if patient_stats and total_apnea_events > 0:
        max_patient = max(patient_stats.items(), key=lambda x: x[1]['total_events'])
        min_patient = min([p for p in patient_stats.items() if p[1]['total_events'] > 0], 
                         key=lambda x: x[1]['total_events'], default=None)
        
        print()
        print(f"Patient with most apnea events: {max_patient[0]} ({max_patient[1]['total_events']} events)")
        if min_patient:
            print(f"Patient with least apnea events: {min_patient[0]} ({min_patient[1]['total_events']} events)")
    
    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # File-level CSV
    df_files = pd.DataFrame(apnea_stats)
    csv_path_files = f"apnea_analysis_files_{timestamp}.csv"
    df_files.to_csv(csv_path_files, index=False)
    print(f"\n💾 File-level results saved to: {csv_path_files}")
    
    # Patient-level CSV
    patient_data = []
    for patient_id, stats in patient_stats.items():
        row = {
            'patient_id': patient_id,
            'files': stats['files'],
            'total_events': stats['total_events'],
            'total_duration_minutes': stats['total_duration'] / 60,
            'unique_event_types': len(stats['event_types'])
        }
        # Add individual event type counts
        for event_type, count in stats['event_types'].items():
            row[f"count_{event_type}"] = count
        
        patient_data.append(row)
    
    df_patients = pd.DataFrame(patient_data)
    csv_path_patients = f"apnea_analysis_patients_{timestamp}.csv"
    df_patients.to_csv(csv_path_patients, index=False)
    print(f"💾 Patient-level results saved to: {csv_path_patients}")
    
    # Print potential issues
    print("\n⚠️ POTENTIAL ISSUES TO INVESTIGATE:")
    print("-" * 50)
    
    if files_without_apnea == len(rml_files):
        print("• ALL files have NO apnea events - this suggests:")
        print("  - Wrong XML parsing method")
        print("  - Events are named differently (not containing 'apnea')")
        print("  - Corrupt or different file format")
    elif files_without_apnea > 0:
        print(f"• {files_without_apnea} files have NO apnea events")
    
    print("\n✅ Analysis complete!")
    return apnea_stats, patient_stats, all_event_names

def debug_rml_structure(rml_path):
    """Debug function to examine RML file structure"""
    try:
        tree = ET.parse(rml_path)
        root = tree.getroot()
        
        print(f"Root tag: {root.tag}")
        print(f"Root attributes: {root.attrib}")
        
        # Try different namespaces
        namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
        
        # Look for Event elements
        events = root.findall('.//ns:Event', namespace)
        print(f"Found {len(events)} Event elements")
        
        if events:
            print("First few events:")
            for i, event in enumerate(events[:5]):
                event_type = event.get('Type')
                start = event.get('Start')
                duration = event.get('Duration')
                print(f"  Event {i}: Type='{event_type}', Start={start}, Duration={duration}")
        
        # Also try without namespace
        events_no_ns = root.findall('.//Event')
        print(f"Found {len(events_no_ns)} Event elements (no namespace)")
        
        # Look for all unique element tags
        all_tags = set()
        for elem in root.iter():
            all_tags.add(elem.tag)
        
        print(f"All unique tags in file: {sorted(list(all_tags))[:20]}")  # First 20
        
    except Exception as e:
        print(f"Error debugging {rml_path}: {e}")

def main():
    """Main function to run the exploration"""
    # Update this path to your data directory
    base_directory = "/raid/userdata/cybulskm/ThesisProj/dataset"
    
    if not os.path.exists(base_directory):
        print(f"❌ Directory not found: {base_directory}")
        print("Please update the base_directory path in the script")
        return
    
    print(f"Starting exploration of: {base_directory}")
    print()
    
    apnea_stats, patient_stats, all_event_names = explore_all_rml_files(base_directory)
    
    # Additional analysis
    print("\n🔬 DETAILED ANALYSIS:")
    print("-" * 30)
    
    # Calculate statistics
    event_counts = [stats['total_apnea_events'] for stats in apnea_stats]
    if event_counts:
        print(f"Event count statistics:")
        print(f"  Mean: {sum(event_counts)/len(event_counts):.1f}")
        print(f"  Min: {min(event_counts)}")
        print(f"  Max: {max(event_counts)}")
        print(f"  Files with 0 events: {event_counts.count(0)}")
        print(f"  Files with >100 events: {sum(1 for x in event_counts if x > 100)}")

if __name__ == "__main__":
    main()