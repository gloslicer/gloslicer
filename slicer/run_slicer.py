import os
import sys
import argparse
import subprocess
import time
import json
from pathlib import Path
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_all_projects(dataset_dir, splits):
    all_projects = []
    for split in splits:
        split_dir = Path(dataset_dir) / split
        if not split_dir.exists():
            continue
        for project in os.listdir(split_dir):
            if (split_dir / project).is_dir():
                all_projects.append( (split, project) )
    return all_projects

def process_project_wrapper(args_tuple, dataset_dir, output_dir, slicer_path, additional_args, timeout):
    split, project = args_tuple
    input_dir = str(Path(dataset_dir) / split)
    output_dir_split = str(Path(output_dir) / split)
    return process_project(project, input_dir, output_dir_split, slicer_path, additional_args, timeout)

def run_slicer_on_directory(input_dir, output_dir, slicer_path, additional_args=None, timeout=300, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    projects = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not projects:
        logger.warning(f"No projects found in {input_dir}")
        return
    logger.info(f"Found {len(projects)} projects in {input_dir}")

    success_count = 0
    timeout_count = 0
    error_count = 0
    total_time = 0

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(projects), desc="Processing projects", unit="project") as pbar:
        for project in projects:
            future = executor.submit(
                process_project, project, input_dir, output_dir, slicer_path, additional_args, timeout
            )
            futures.append(future)
        for f in as_completed(futures):
            project, status, processing_time, error_msg = f.result()
            if status == 'success':
                success_count += 1
                total_time += processing_time
            elif status == 'timeout':
                timeout_count += 1
                logger.warning(f"Timeout ({timeout}s) reached for project {project}")
            else:
                error_count += 1
                logger.error(f"Error processing {project}: {error_msg}")
            avg_time = total_time / max(1, success_count)
            pbar.set_postfix({"avg": f"{avg_time:.1f}s", "success": success_count, "timeout": timeout_count, "error": error_count})
            pbar.update(1)
    logger.info(f"Summary: {success_count} successful, {timeout_count} timeout, {error_count} error")
    if success_count > 0:
        logger.info(f"Average processing time: {total_time/success_count:.2f} seconds per project")

def merge_slices(slices_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(slices_dir, split)
        if not os.path.exists(split_dir):
            logger.warning(f"{split_dir} directory doesn't exist, skipping")
            continue
        
        projects = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        if not projects:
            logger.warning(f"No projects found in {split_dir}, skipping")
            continue
        
        logger.info(f"Merging {len(projects)} projects from {split} split")
        
        all_slices = []
        
        for project in tqdm(projects, desc=f"Merging {split} data"):
            slices_json_path = os.path.join(split_dir, project, "slices.json")
            
            if not os.path.exists(slices_json_path):
                logger.warning(f"Skipping {project}: {slices_json_path} doesn't exist")
                continue
            
            try:
                with open(slices_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for slice_data in data:
                        slice_data['project'] = project
                    
                    all_slices.extend(data)
                    
                    logger.debug(f"Loaded {len(data)} slices from {project}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {slices_json_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing {slices_json_path}: {e}")
        
        output_file = os.path.join(output_dir, f"{split}.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_slices, f, ensure_ascii=False)
            
            logger.info(f"Saved {len(all_slices)} slices to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save to {output_file}: {e}")
            
def process_project(project, input_dir, output_dir, slicer_path, additional_args, timeout):
    project_input_path = os.path.join(input_dir, project)
    project_output_path = os.path.join(output_dir, project)
    os.makedirs(project_output_path, exist_ok=True)
    output_json = os.path.join(project_output_path, "slices.json")
    cmd = [sys.executable, slicer_path, '-d', project_input_path, '-o', output_json]
    if additional_args:
        cmd.extend(additional_args)
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        processing_time = time.time() - start_time
        return (project, 'success', processing_time, None)
    except subprocess.TimeoutExpired:
        return (project, 'timeout', timeout, None)
    except subprocess.CalledProcessError as e:
        return (project, 'error', time.time() - start_time, e.stderr)
    except Exception as e:
        return (project, 'error', time.time() - start_time, str(e))

def main():
    parser = argparse.ArgumentParser(description='Run slicer.py on Python projects in dataset directories')
    parser.add_argument('-d', '--dataset', default='./dataset', help='Dataset directory containing train, val, test subdirectories (default: dataset)')
    parser.add_argument('-o', '--output', default='./slices', help='Output directory for slicing results (default: slices)')
    parser.add_argument('-s', '--slicer', default='./slicer/slicer.py', help='Path to slicer.py script (default: ./slicer/slicer.py)')
    parser.add_argument('--additional-args', nargs=argparse.REMAINDER, help='Additional arguments to pass to slicer.py')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-t', '--timeout', type=int, default=36000, help='Timeout in seconds for processing each project (default: 36000 seconds)')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge slice results after processing')
    parser.add_argument('--merge-dir', default='./model/data', help='Directory to store merged slice files (default: ./model/data)')
    parser.add_argument('--workers', type=int, default=16, help='Maximum number of parallel slicing processes')

    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        logger.error(f"Dataset directory {args.dataset} does not exist")
        sys.exit(1)
    
    slicer_path = Path(args.slicer)
    if not slicer_path.exists():
        logger.error(f"Slicer script {args.slicer} does not exist")
        sys.exit(1)
    
    splits = ['train', 'val', 'test']
    total_splits = sum(1 for split in splits if (dataset_dir / split).exists())
        
    splits = ['train', 'val', 'test']
    all_projects = get_all_projects(args.dataset, splits)
    logger.info(f"Total projects to process: {len(all_projects)}")
    
    success_count = 0
    timeout_count = 0
    error_count = 0
    total_time = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor, tqdm(total=len(all_projects), desc="Processing all projects", unit="project") as pbar:
        futures = [
            executor.submit(
                process_project_wrapper,
                tup, args.dataset, args.output, str(args.slicer), args.additional_args, args.timeout
            ) for tup in all_projects
        ]
        for f in as_completed(futures):
            project, status, processing_time, error_msg = f.result()
            if status == 'success':
                success_count += 1
                total_time += processing_time
            elif status == 'timeout':
                timeout_count += 1
                logger.warning(f"Timeout ({args.timeout}s) reached for project {project}")
            else:
                error_count += 1
                logger.error(f"Error processing {project}: {error_msg}")
            avg_time = total_time / max(1, success_count)
            pbar.set_postfix({"avg": f"{avg_time:.1f}s", "success": success_count, "timeout": timeout_count, "error": error_count})
            pbar.update(1)
    logger.info(f"Summary: {success_count} successful, {timeout_count} timeout, {error_count} errors")
    if success_count > 0:
        logger.info(f"Average processing time: {total_time/success_count:.2f} seconds per project")

    logger.info("Slicing completed successfully")

    if args.merge:
        logger.info("Merging slice results...")
        merge_slices(args.output, args.merge_dir)
        logger.info(f"Merged results saved to {args.merge_dir}")

if __name__ == "__main__":
    main() 