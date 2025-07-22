import pickle
import os
import numpy as np
from collections import Counter

dataset_dir = './model/dataset/longformer-large'
splits = ['train', 'val', 'test']
label_names = ['none', 'forward', 'backward']

def analyze_split(path):
    with open(path, 'rb') as f:
        data_list = pickle.load(f)
    num_samples = len(data_list)
    num_nodes_list = [data.x.shape[0] for data in data_list]
    avg_nodes = np.mean(num_nodes_list)

    all_labels = []
    for data in data_list:
        # data.y shape: [num_nodes]
        all_labels.extend(data.y.cpu().numpy().tolist())
    label_counter = Counter(all_labels)
    total_labels = sum(label_counter.values())
    label_dist = {label_names[i]: f"{label_counter.get(i, 0)} ({label_counter.get(i, 0) / total_labels:.2%})" for i in range(3)}
    return num_samples, avg_nodes, label_dist

print(f"{'Split':<8} {'#Samples':<10} {'Avg. Nodes/Graph':<18} {'Label Distribution'}")
for split in splits:
    pkl_path = os.path.join(dataset_dir, f"{split}.pkl")
    num_samples, avg_nodes, label_dist = analyze_split(pkl_path)
    label_dist_str = ', '.join([f"{k}: {v}" for k, v in label_dist.items()])
    print(f"{split.capitalize():<8} {num_samples:<10} {avg_nodes:<18.2f} {label_dist_str}")
