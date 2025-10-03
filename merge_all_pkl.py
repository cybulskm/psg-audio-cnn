
import os

def merge_pickles(pkl_dir, output_file):
    import pickle

    merged_data = []
    for pkl_file in os.listdir(pkl_dir):
        if pkl_file.endswith('.pkl'):
            with open(os.path.join(pkl_dir, pkl_file), 'rb') as f:
                data = pickle.load(f)
                merged_data.extend(data)

    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f)


if __name__ == "__main__":
    pkl_dir = 'bin'
    output_file = 'merged_data.pkl'
    merge_pickles(pkl_dir, output_file)