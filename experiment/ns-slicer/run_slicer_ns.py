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

def run_one_slicer_job(cmd, project, split, timeout):
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
        return 'success', time.time() - t0
    except subprocess.TimeoutExpired:
        return 'timeout', timeout
    except subprocess.CalledProcessError as e:
        return 'error', e.stderr
    except Exception as e:
        return 'error', str(e)

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
            except Exception as e:
                logger.error(f"Error processing {slices_json_path}: {e}")
        output_file = os.path.join(output_dir, f"{split}.json")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_slices, f, ensure_ascii=False)
            logger.info(f"Saved {len(all_slices)} slices to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save to {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run slicer.py on all dataset projects (train/val/test) in parallel, and support ablation/context modes.')
    parser.add_argument('-d', '--dataset', default='./dataset', help='Dataset directory containing train, val, test subdirectories')
    parser.add_argument('-o', '--output', default='./experiment/slices', help='Output directory for slicing results')
    parser.add_argument('-s', '--slicer', default='./experiment/ns-slicer/slicer_ns_slicer.py', help='Path to slicer_ns_slicer.py script')
    parser.add_argument('--additional-args', nargs=argparse.REMAINDER, help='Additional arguments to pass to slicer.py')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-t', '--timeout', type=int, default=3600, help='Timeout in seconds for processing each project')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge slice results after processing')
    parser.add_argument('--merge-dir', default='./experiment/model/data', help='Directory to store merged slice files')
    parser.add_argument('--workers', type=int, default=16, help='Max parallel slicing processes')
    parser.add_argument('--input_modes', nargs='+', default=['func_only', 'func_plus_called', 'func_plus_caller', 'func_plus_all'],
                        help='Slicing input modes to run, e.g. func_only func_plus_called ... (default: all)')
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

    for input_mode in args.input_modes:
        logger.info(f"=== Start slicing mode: {input_mode} ===")
        mode_output_dir = Path(args.output) / input_mode
        tasks = []
        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                logger.warning(f"{split} directory not found in {args.dataset}")
                continue
            projects = [d for d in os.listdir(split_dir) if os.path.isdir(split_dir / d)]
            logger.info(f"[{input_mode}] Found {len(projects)} projects in {split} split")
            for project in projects:
                project_input_path = str(split_dir / project)
                project_output_path = str(mode_output_dir / split / project)
                os.makedirs(project_output_path, exist_ok=True)
                output_json = os.path.join(project_output_path, "slices.json")
                tasks.append({
                    "project": project,
                    "split": split,
                    "input_path": project_input_path,
                    "output_json": output_json,
                    "output_dir": project_output_path,
                    "input_mode": input_mode,
                })
        logger.info(f"[{input_mode}] Total {len(tasks)} projects to process")
        with ProcessPoolExecutor(max_workers=args.workers) as executor, tqdm(total=len(tasks), desc=f"Slicing ({input_mode})", unit="proj") as pbar:
            future_to_task = {}
            for t in tasks:
                cmd = [
                    sys.executable,
                    str(slicer_path),
                    '-d', t["input_path"],
                    '-o', t["output_json"],
                    '--input_mode', t["input_mode"],
                    '--split', t["split"]
                ]
                if args.additional_args:
                    cmd.extend(args.additional_args)
                future = executor.submit(run_one_slicer_job, cmd, t["project"], t["split"], args.timeout)
                future_to_task[future] = t
            for future in as_completed(future_to_task):
                t = future_to_task[future]
                try:
                    status, info = future.result()
                    if status == 'success':
                        logger.info(f"[{input_mode}][{t['split']}] {t['project']} done in {info:.1f}s")
                    elif status == 'timeout':
                        logger.warning(f"[{input_mode}][{t['split']}] {t['project']} timeout")
                    else:
                        logger.error(f"[{input_mode}][{t['split']}] {t['project']} error: {info}")
                except Exception as e:
                    logger.error(f"[{input_mode}][{t['split']}] {t['project']} exception: {str(e)}")
                pbar.update(1)
        if args.merge:
            logger.info(f"Merging slice results for {input_mode} ...")
            merge_slices(str(mode_output_dir), str(Path(args.merge_dir) / input_mode))
            logger.info(f"Merged results saved to {Path(args.merge_dir) / input_mode}")

    logger.info("All slicing jobs finished.")

if __name__ == "__main__":
    main()