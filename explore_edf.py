import pyedflib
import numpy as np
from collections import Counter

def get_most_common_frequency(file_path):
    """Return the most common sample frequency and the labels that have it."""
    with pyedflib.EdfReader(file_path) as f:
        headers = f.getSignalHeaders()
        freqs = [h['sample_frequency'] for h in headers]
        # most common frequency
        common_freq, _ = Counter(freqs).most_common(1)[0]
        # labels for that frequency
        labels = [h['label'] for h in headers if h['sample_frequency'] == common_freq]
    return common_freq, labels

def explore_edf(file_path, labels_to_read):
    """Read only channels whose labels are in labels_to_read."""
    with pyedflib.EdfReader(file_path) as f:
        # find channel indices for these labels
        headers = f.getSignalHeaders()
        indices = [i for i,h in enumerate(headers) if h['label'] in labels_to_read]
        n_channels = len(indices)

        # figure out max samples per channel
        max_samples = max(f.getNSamples()[i] for i in indices)
        sigbufs = np.zeros((n_channels, max_samples))

        for idx, ch in enumerate(indices):
            n_samples = f.getNSamples()[ch]
            signal = f.readSignal(ch)
            sigbufs[idx, :n_samples] = signal

        return sigbufs, [headers[i]['label'] for i in indices]


def preview_edf(file_path, labels_to_read=None, start=0, n_samples=1000):
    with pyedflib.EdfReader(file_path) as f:
        headers = f.getSignalHeaders()

        # Pick all channels if labels_to_read is None
        if labels_to_read is None:
            indices = list(range(f.signals_in_file))
        else:
            indices = [i for i,h in enumerate(headers) if h['label'] in labels_to_read]

        # read a chunk from each selected channel
        chunks = []
        for idx in indices:
            # Make sure we don't exceed channel length
            total_samples = f.getNSamples()[idx]
            n = min(n_samples, total_samples - start)
            data_chunk = f.readSignal(idx, start=start, n=n)
            chunks.append(data_chunk)

        # Return as a NumPy array (channels x n_samples)
        max_len = max(len(c) for c in chunks)
        arr = np.zeros((len(chunks), max_len))
        for i,c in enumerate(chunks):
            arr[i,:len(c)] = c

        labels = [headers[i]['label'] for i in indices]
    return arr, labels


def read_chunk_from_data(file_path, labels_to_read=None, start=0, n_samples=1000):
    with pyedflib.EdfReader(file_path) as f:
        headers = f.getSignalHeaders()

        # Pick all channels if labels_to_read is None
        if labels_to_read is None:
            indices = list(range(f.signals_in_file))
        else:
            indices = [i for i,h in enumerate(headers) if h['label'] in labels_to_read]

        # read a chunk from each selected channel
        chunks = []
        for idx in indices:
            # Make sure we don't exceed channel length
            total_samples = f.getNSamples()[idx]
            n = min(n_samples, total_samples - start)
            data_chunk = f.readSignal(idx, start=start, n=n)
            chunks.append(data_chunk)

        # Return as a NumPy array (channels x n_samples)
        max_len = max(len(c) for c in chunks)
        arr = np.zeros((len(chunks), max_len))
        for i,c in enumerate(chunks):
            arr[i,:len(c)] = c

        labels = [headers[i]['label'] for i in indices]
    return arr, labels


def print_labels_and_freqs(file_path):
    with pyedflib.EdfReader(file_path) as f:
        headers = sorted(f.getSignalHeaders(), key=lambda h: h['sample_frequency'], reverse=True)
        # group by frequency
        freq_groups = {}
        for h in headers:
            freq = h['sample_frequency']
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(h['label'])
        for freq, labels in sorted(freq_groups.items(), reverse=True):
            print(f"Frequency: {freq} Hz\nChannels: {labels}")
            print("---------------")

# Usage
file_path = 'data\\00000995-100507[001].edf'

print_labels_and_freqs(file_path)
"""
common_freq, labels = get_most_common_frequency(file_path)
arr, labels = preview_edf(file_path, start=0, n_samples=500)
print("Labels:", labels)
print("Data shape (channels x samples):", arr.shape)
print("First channel first 10 samples:", arr[0,:10])
"""