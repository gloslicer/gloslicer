import os
import pandas as pd
import pickle
from collections import defaultdict

base_dir = './model/dataset/longformer-large'
splits = ['train.pkl', 'val.pkl', 'test.pkl']
suspect_csv_path = './cross_verification/suspect_samples.csv'

id2label = {0: 'None Slice', 1: 'Forward Slice', 2: 'Backward Slice'}

df_suspect = pd.read_csv(suspect_csv_path)
suspect_node_ids = set((row['sample_id'], row['node_idx']) for _, row in df_suspect.iterrows())

total_count = defaultdict(int)
suspect_count = defaultdict(int)

global_sample_id = 0
for split in splits:
    path = os.path.join(base_dir, split)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        continue

    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    for sample in dataset:
        if not hasattr(sample, 'y'):
            global_sample_id += 1
            continue
        for node_idx, label_id in enumerate(sample.y.tolist()):
            label_name = id2label.get(label_id, 'Unknown')
            total_count[label_name] += 1
            if (global_sample_id, node_idx) in suspect_node_ids:
                suspect_count[label_name] += 1
        global_sample_id += 1

print("\nAccuracy Table:")
print(f"{'Label Type':<20} {'Accuracy (%)':>15}")
print("-" * 35)
rows = []
for label in id2label.values():
    total = total_count[label]
    suspect = suspect_count[label]
    acc = 100.0 * (1 - suspect / total) if total > 0 else 0.0
    print(f"{label:<20} {acc:>15.2f}")
    rows.append({"Label": label, "Accuracy (%)": acc})
