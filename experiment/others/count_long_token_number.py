import os
import json
import pickle
import sys
sys.path.append("/fred/oz339/siyu/lslicer/experiment/ns-slicer/src")
import utils

analysis_result_dir = "./experiment/model/dataset/truncate"
split_names = ['train', 'val', 'test']

def count_long_token_number():
    long_token_count = 0
    total_count = 0
    for split in split_names:
        stats_path = os.path.join(analysis_result_dir, f"{split}_long_code_stats.json")
        pkl_path = os.path.join(analysis_result_dir, f"{split}_truncate.pkl")
        if not os.path.exists(stats_path):
            print(f"Warning: {stats_path} does not exist.")
            long_code_num = 0
        else:
            with open(stats_path, 'r') as f:
                long_code_data = json.load(f)
                long_code_num = len(long_code_data)
        long_token_count += long_code_num

        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} does not exist.")
            split_total = 0
        else:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                split_total = len(data)
        total_count += split_total

        print(f"{split} : Long code {long_code_num} / Total samples {split_total}")

    return long_token_count, total_count

if __name__ == "__main__":
    number, total = count_long_token_number()
    if total > 0:
        percent = number / total * 100
        print(f"\nTotal long code samples: {number} / {total} ({percent:.2f}%)")
    else:
        print("No samples found!")

