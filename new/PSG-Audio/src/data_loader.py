# data_loader.py

import os
import pickle
import numpy as np

def load_data(data_path, channels):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        segments = data
    else:
        segments = [data]
        try:
            while True:
                segment = pickle.load(f)
                segments.append(segment)
        except EOFError:
            pass
    
    X_all = []
    y_all = []
    
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        
        channel_data = []
        valid = True
        
        for ch in channels:
            if ch in seg and seg[ch] is not None:
                data = np.array(seg[ch], dtype=np.float32)
                if len(data) > 0:
                    channel_data.append(data)
                else:
                    valid = False
                    break
            else:
                valid = False
                break
        
        if valid and len(channel_data) == len(channels):
            min_len = min(len(ch) for ch in channel_data)
            if min_len > 10:
                channel_data = [ch[:min_len] for ch in channel_data]
                X_all.append(np.array(channel_data).T)
                y_all.append(seg.get('Label', 'Unknown'))
    
    if not X_all:
        raise ValueError("No valid segments processed!")
    
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all)
    
    return X, y

def preprocess_data(X):
    X_std = np.copy(X)
    for i in range(X.shape[-1]):
        channel_data = X[:, :, i]
        mean = np.mean(channel_data)
        std = np.std(channel_data)
        if std > 0:
            X_std[:, :, i] = (channel_data - mean) / std
    return X_std