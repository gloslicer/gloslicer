import os
import sys
import argparse
import shutil
import random
from collections import defaultdict

def split_data(data_dir, output_dir):
    project_files = defaultdict(list)
    for root, dirs, files in os.walk(data_dir):
        rel_path = os.path.relpath(root, data_dir)
        if rel_path == '.':
            continue
        project = rel_path.split(os.sep)[0]
        for filename in files:
            if filename.endswith('.py'):
                rel_file = os.path.join(rel_path, filename)
                project_files[project].append(rel_file)

    projects = list(project_files.keys())
    random.shuffle(projects)
    
    total_projects = len(projects)
    train_size = int(total_projects * 0.6)
    val_size = int(total_projects * 0.2)
    
    train_projects = projects[:train_size]
    val_projects = projects[train_size:train_size + val_size]
    test_projects = projects[train_size + val_size:]
    
    splits = {
        'train': train_projects,
        'val': val_projects,
        'test': test_projects
    }
    
    for split_name, split_projects in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        total_files = 0
        
        for project in split_projects:
            dest_proj_dir = os.path.join(split_dir, project)
            os.makedirs(dest_proj_dir, exist_ok=True)
            for rel_file in project_files[project]:
                src = os.path.join(data_dir, rel_file)
                dst = os.path.join(dest_proj_dir, os.path.basename(rel_file))
                shutil.copy2(src, dst)
            total_files += len(project_files[project])
            
        print(f'[+] {split_name}: {len(split_projects)} projects ({len(split_projects)/total_projects:.1%}), {total_files} files → {split_dir}')
        
def run_slicer(data_dir, output_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            output_split_dir = os.path.join(output_dir, split)
            os.makedirs(output_split_dir, exist_ok=True)
            print(f"Running slicer on {split_dir} -> {output_split_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir',    dest='root',   required=True,
                        help="Dataset root directory containing project subdirectories")
    parser.add_argument('-s', '--save', dest='save_dir', default='./dataset',
                        help="Output directory where the script will create train, val, test subdirectories with 60:20:20 project ratio")
    args = parser.parse_args()
    split_data(args.root, args.save_dir)

if __name__ == "__main__":
    main()
