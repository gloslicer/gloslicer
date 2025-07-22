import os
import json
import subprocess
import logging
import argparse
import time
from pathlib import Path
import concurrent.futures
import shutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clone_repository(repo_info, output_dir, max_retries=3):
    repo_name = repo_info["name"]
    repo_url = repo_info["url"]
    
    target_dir = os.path.join(output_dir, repo_name)
    
    if os.path.exists(target_dir):
        logger.info(f"Repository {repo_name} already exists, skipping")
        return False
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Cloning repository: {repo_name} -> {target_dir} (attempt {attempt+1}/{max_retries})")
            subprocess.run(
                ["git", "clone", "--depth=1", repo_url, target_dir],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300
            )
            logger.info(f"Successfully cloned repository: {repo_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository {repo_name}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to clone repository {repo_name}, maximum retries reached")
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir, ignore_errors=True)
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Cloning repository {repo_name} timed out")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir, ignore_errors=True)
            return False
    
    return False

def run_crawler(crawler_script=None):
    if crawler_script is None:
        crawler_script = os.path.join("crawler", "crawler.py")
    
    try:
        logger.info(f"Running crawler script: {crawler_script}")
        
        subprocess.run(
            ["python", crawler_script],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run crawler script: {e}")
        return False

def check_disk_space(min_required_gb=10):
    # Get available disk space in bytes
    if os.name == 'nt':  # Windows
        free_bytes = shutil.disk_usage('.').free
    else:  # Linux/Mac
        stat = os.statvfs('.')
        free_bytes = stat.f_bavail * stat.f_frsize
    
    free_gb = free_bytes / (1024 ** 3)  # Convert to GB
    
    if free_gb < min_required_gb:
        logger.warning(f"Low disk space! Only {free_gb:.2f} GB available, recommended at least {min_required_gb} GB")
        return False
    
    logger.info(f"Disk space check passed: {free_gb:.2f} GB available")
    return True

def main():
    parser = argparse.ArgumentParser(description='Clone GitHub repositories or run crawler')
    parser.add_argument('--data-dir', default='data_raw', help='Data directory')
    parser.add_argument('--max-repos', type=int, default=None, help='Maximum number of repositories to clone')
    parser.add_argument('--crawler-script', default=None, help='Crawler script path, default is crawler/crawler.py')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers for cloning')
    args = parser.parse_args()
    
    data_raw_dir = args.data_dir
    json_path = os.path.join(data_raw_dir, "python_projects.json")
    
    if not os.path.exists(data_raw_dir):
        logger.info(f"Creating directory: {data_raw_dir}")
        os.makedirs(data_raw_dir)
    
    if not check_disk_space(min_required_gb=10):
        logger.warning("Disk space may be insufficient, but continuing anyway")
    
    if os.path.exists(json_path):
        logger.info(f"Found projects list file: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                repos = json.load(f)
            
            total_repos = len(repos)
            logger.info(f"Loaded {total_repos} repository information")
            
            if args.max_repos and args.max_repos < total_repos:
                logger.info(f"Limiting clone count to first {args.max_repos} repositories")
                repos = repos[:args.max_repos]
            
            progress_bar = tqdm(total=len(repos), desc="clone repo status", unit="repo")
            
            successful_clones = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(clone_repository, repo, data_raw_dir): repo for repo in repos}
                for future in concurrent.futures.as_completed(futures):
                    repo = futures[future]
                    try:
                        if future.result():
                            successful_clones += 1
                    except Exception as e:
                        logger.error(f"Error processing repository {repo['name']}: {e}")
                    progress_bar.update(1)
            
            progress_bar.close()
            
            logger.info(f"Successfully cloned {successful_clones}/{len(repos)} repositories")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {json_path}, file may not be valid JSON")
            run_crawler(args.crawler_script)
        
        except Exception as e:
            logger.error(f"Error processing {json_path}: {e}")
            run_crawler(args.crawler_script)
    
    else:
        logger.info(f"Project list file not found: {json_path}")
        run_crawler(args.crawler_script)

if __name__ == "__main__":
    main()