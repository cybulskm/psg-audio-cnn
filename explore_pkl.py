
import pickle
from collections import Counter
import matplotlib.pyplot as plt

def explore_pkl(file_path):
    # Load the .pkl file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if the data is a list of dictionaries
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        # Extract the 'Label' values
        labels = [item['Label'] for item in data if 'Label' in item]
        
        # Count the occurrences of each unique label
        label_counts = Counter(labels)
        
        # Print the occurrences of each unique label
        print("Occurrences of each unique label:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")
            

    else:
        print("The data is not in the expected format (list of dictionaries with 'Label' key).")


def visualize_psg_pkl(file_path, max_items=1):
    """
    Visualize PSG signals from a pkl file where each entry is a dict of channels + 'Label'.
    
    Parameters
    ----------
    file_path : str
        Path to the .pkl file
    max_items : int
        Number of samples to visualize per label (to avoid huge plots)
    """
    # Load
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        labels = [item['Label'] for item in data if 'Label' in item]
        label_counts = Counter(labels)
        print("Occurrences of each unique label:")
        for label, count in label_counts.items():
            print(f"{label}: {count}")

        # Figure out which keys are channels (everything except 'Label')
        all_keys = set(data[0].keys())
        all_keys.discard('Label')
        channel_keys = list(all_keys)
        print("Channels found:", channel_keys) 

        # Plot signals per label
        for label in label_counts:
            # Filter items with this label
            items = [item for item in data if item.get('Label') == label]

            # Limit how many you show
            items = items[:max_items]

            for idx, item in enumerate(items, start=1):
                plt.figure(figsize=(12, 1.5*len(channel_keys)))  # slightly shorter per channel
                for ch_i, ch_key in enumerate(channel_keys, start=1):
                    plt.subplot(len(channel_keys), 1, ch_i)
                    y = item[ch_key]
                    plt.plot(y, linewidth=0.8)
                    plt.ylabel(ch_key, fontsize=7)         
                    plt.tick_params(axis='both', labelsize=7) 
                    plt.grid(True)

                plt.tight_layout(rect=[0, 0.01, 1, 0.95])
                plt.subplots_adjust(hspace=0.5)  # more vertical space between subplots
                plt.suptitle(f"Label: {label} (sample {idx})", fontsize=10)
                plt.show()
    else:
        print("The data is not in the expected format (list of dictionaries with 'Label' key).")

# Path to the .pkl file
pkl_file = 'processed.pkl'

# Explore the .pkl file
#explore_pkl(pkl_file)

visualize_psg_pkl(pkl_file)