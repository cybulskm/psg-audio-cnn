import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def draw():
    # Load the data
    with open('single_channel.pkl', 'rb') as f:
        data = pickle.load(f)

    # Create time array
    time_array = np.arange(len(data['signal'])) / data['sampling_rate']

    plt.figure(figsize=(15, 8))

    # Plot the full signal (normal sleep)
    plt.plot(time_array, data['signal'], 'b-', alpha=0.7, linewidth=0.5, label='EEG C3-A2 (Normal Sleep)')

    # Get unique apnea event types
    apnea_events = np.unique([ann[0] for ann in data['annotations']])
    print(f"Unique apnea events: {apnea_events}")

    # Create color map for different apnea types
    colors = ['red', 'orange', 'purple', 'green', 'brown'][:len(apnea_events)]
    color_map = dict(zip(apnea_events, colors))

    # Highlight apnea events
    legend_handles = []
    for event_type in apnea_events:
        # Plot all events of this type
        for ann in data['annotations']:
            if ann[0] == event_type:
                start_time, duration = ann[1], ann[2]
                end_time = start_time + duration
                plt.axvspan(start_time, end_time, alpha=0.5, color=color_map[event_type])
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=color_map[event_type], alpha=0.5, label=event_type))
    for item in data["annotations"]:
        print(item)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('EEG C3-A2 Signal with Apnea Events Highlighted\n(Normal Sleep = Blue, Apnea Events = Colored Regions)')
    plt.legend(handles=legend_handles, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nSummary:")
    print(f"Total recording time: {data['total_duration']:.2f} seconds ({data['total_duration']/3600:.2f} hours)")
    print(f"Number of apnea events: {len(data['annotations'])}")
    for event_type in apnea_events:
        count = sum(1 for ann in data['annotations'] if ann[0] == event_type)
        print(f"  {event_type}: {count} events")


def write():
    import os
    import pickle
    import numpy as np
    import pyedflib
    import xml.etree.ElementTree as ET

        
    def parse_annotations(rml_file_path):
        """Parse apnea events from the .rml file."""
        tree = ET.parse(rml_file_path)
        root = tree.getroot()
        namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}
        events = []
        for event in root.findall('.//ns:Event', namespace):
            event_type = event.get('Type')
            start_time = float(event.get('Start'))
            duration = float(event.get('Duration'))
            # Only include apnea events
            if 'apnea' in event_type.lower():
                events.append((event_type, start_time, duration))
        return events

    def extract_channel_from_edf(edf_file_path, channel_name):
        """Extract raw data for one channel and its sampling rate."""
        edf_file = pyedflib.EdfReader(edf_file_path)
        signal_labels = edf_file.getSignalLabels()
        if channel_name not in signal_labels:
            raise ValueError(f"{channel_name} not found in {edf_file_path}")
        idx = signal_labels.index(channel_name)
        signal_data = edf_file.readSignal(idx)
        sampling_rate = edf_file.getSampleFrequency(idx)
        edf_file.close()
        return signal_data, sampling_rate

    def save_channel_and_annotations(edf_files, rml_file, channel_name, output_pkl):
        """Concatenate channel across EDFs and save with annotations."""
        annotations = parse_annotations(rml_file)
        print(f"Found {len(annotations)} apnea events in {rml_file}")

        full_signal = []
        sampling_rate = None
        file_start_times = []  # Track absolute start time of each EDF file
        current_start_time = 0.0

        # Sort EDF files to ensure correct order
        edf_files.sort()
        
        # First pass: extract all signals and track file timing
        for edf_file in edf_files:
            print(f"Processing {edf_file}...")
            sig, sr = extract_channel_from_edf(edf_file, channel_name)
            if sampling_rate is None:
                sampling_rate = sr
            elif sr != sampling_rate:
                raise ValueError("All EDF files must have the same sampling rate")
            
            full_signal.extend(sig)
            file_duration = len(sig) / sr
            file_start_times.append(current_start_time)
            current_start_time += file_duration

        full_signal = np.array(full_signal)
        total_duration = len(full_signal) / sampling_rate
        print(f"Total signal duration: {total_duration:.2f} seconds")
        
        # Keep annotations with absolute timing (no adjustment needed)
        # The annotations already have absolute timing from the start of the recording
        adjusted_annotations = []
        for event_type, start_time, duration in annotations:
            # Only include events that occur within the total recording duration
            if start_time < total_duration:
                adjusted_annotations.append((event_type, start_time, duration))
            else:
                print(f"Warning: Event at {start_time}s exceeds total duration {total_duration}s")

        print(f"Number of apnea events within recording: {len(adjusted_annotations)}")

        with open(output_pkl, 'wb') as f:
            pickle.dump({
                'channel_name': channel_name,
                'sampling_rate': sampling_rate,
                'signal': full_signal,
                'annotations': adjusted_annotations,
                'total_duration': total_duration,
                'file_start_times': file_start_times  # For reference
            }, f)
        print(f"Saved {channel_name} signal and annotations to {output_pkl}")

    # Example usage
    if __name__ == "__main__":
        # Get all EDF files for this patient
        data_dir = "data"
        edf_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".edf") and "00000995-100507" in file]
        rml_file = os.path.join(data_dir, "00000995-100507.rml")
        
        # Print the files to verify order
        print("EDF files to process (in order):")
        for file in sorted(edf_files):
            print(f"  {file}")
        
        save_channel_and_annotations(edf_files, rml_file,
                                    channel_name="EEG C3-A2",
                                    output_pkl="single_channel.pkl")

write()
draw()