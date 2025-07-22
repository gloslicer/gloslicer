import requests
import json
import os
import time
import random
from datetime import datetime
import logging
import subprocess
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import signal
from contextlib import contextmanager
import glob

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")

    original_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

class GitHubCrawler:
    def __init__(self):
        self.token = os.getenv('GITHUB_TOKEN')
        if not self.token:
            raise ValueError("Set GITHUB_TOKEN environment variable")

        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json',
        }
        self.api_url = 'https://api.github.com/graphql'
        
        self.data_dir = os.path.join(os.getcwd(), 'data_raw')
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Save data to: {self.data_dir}")

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.counter_file = os.path.join(self.data_dir, 'crawler_counter.json')
        self.load_counter()

        self.projects_file = os.path.join(self.data_dir, 'python_projects.json')
        self.load_projects()

    def load_counter(self):
        try:
            if os.path.exists(self.counter_file):
                with open(self.counter_file, 'r') as f:
                    self.counter = json.load(f)
            else:
                self.counter = {
                    'total_checked': 0,
                    'total_cloned': 0,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        except Exception as e:
            logger.error(f"Failed to load counter: {e}")
            self.counter = {
                'total_checked': 0,
                'total_cloned': 0,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def load_projects(self):
        """
        """
        try:
            if os.path.exists(self.projects_file):
                with open(self.projects_file, 'r', encoding='utf-8') as f:
                    self.projects = json.load(f)
            else:
                self.projects = []
        except Exception as e:
            logger.error(f"Failed to load projects: {e}")
            self.projects = []

    def save_counter(self):
        try:
            self.counter['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.counter_file, 'w') as f:
                json.dump(self.counter, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save counter: {e}")

    def save_projects(self):
        try:
            with open(self.projects_file, 'w', encoding='utf-8') as f:
                json.dump(self.projects, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated project data, total {len(self.projects)} projects")
        except Exception as e:
            logger.error(f"Failed to save projects: {e}")

    def count_python_files(self, repo_path):
        try:
            python_files = glob.glob(os.path.join(repo_path, '**', '*.py'), recursive=True)
            return len(python_files)
        except Exception as e:
            logger.error(f"Error counting Python files: {str(e)}")
            return 0
        
    def random_delay(self):
        delay = random.uniform(10, 20)
        logger.info(f"Waiting for {delay:.2f} seconds...")
        time.sleep(delay)

    def clone_repository(self, repo_url, repo_name):
        repo_path = os.path.join(self.data_dir, repo_name)
        
        if os.path.exists(repo_path):
            python_file_count = self.count_python_files(repo_path)
            if 1 <= python_file_count <= 20:
                logger.info(f"Repository {repo_name} already exists, containing {python_file_count} Python files")
                return True
            else:
                logger.info(f"Repository {repo_name} already exists, but the number of Python files ({python_file_count}) does not meet the requirements")
                return False

        try:
            with timeout(300):
                subprocess.run(
                    ['git', 'clone', repo_url, repo_path],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                python_file_count = self.count_python_files(repo_path)
                if 1 <= python_file_count <= 30:
                    logger.info(f"Successfully cloned repository: {repo_name}, containing {python_file_count} Python files")
                    return True
                else:
                    logger.info(f"Repository {repo_name} Python file count ({python_file_count}) does not meet the requirements, deleting repository")
                    subprocess.run(['rm', '-rf', repo_path], check=True)
                    return False
                    
        except TimeoutException:
            logger.error(f"Cloning repository timed out: {repo_name}")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {repo_name}, error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"An unknown error occurred while cloning the repository: {repo_name}, error: {str(e)}")
            return False
        
    def search_python_projects(self, cursor=None):
        query = """
        {
          search(
            query: "language:python created:>2020-01-01 sort:stars"
            type: REPOSITORY
            first: 100
            after: %s
          ) {
            pageInfo {
              hasNextPage
              endCursor
            }
            nodes {
              ... on Repository {
                name
                url
                stargazerCount
                createdAt
                description
              }
            }
          }
        }
        """ % (f'"{cursor}"' if cursor else 'null')

        try:
            with timeout(30):
                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json={'query': query},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
        except TimeoutException:
            logger.error("API request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None

    def crawl(self):
        cursor = None
        page = 1
        min_python_files = 1
        max_python_files = 20

        # 使用tqdm创建进度条
        with tqdm(desc="Crawling", unit="page") as pbar:
            while True:
                logger.info(f"Crawling page {page}...")
                result = self.search_python_projects(cursor)
                
                if not result or 'errors' in result:
                    logger.error("Query failed, stopping crawl")
                    break

                search_data = result.get('data', {}).get('search', {})
                nodes = search_data.get('nodes', [])
                
                if not nodes:
                    break

                for repo in nodes:
                    self.counter['total_checked'] += 1
                    
                    project_info = {
                        'name': repo['name'],
                        'url': repo['url'],
                        'stars': repo['stargazerCount'],
                        'created_at': repo['createdAt'],
                        'description': repo['description']
                    }
                    
                    if self.clone_repository(repo['url'], repo['name']):
                        self.projects.append(project_info)
                        self.counter['total_cloned'] += 1
                        self.save_projects()
                    
                    if self.counter['total_checked'] % 10 == 0:
                        self.save_counter()
                        logger.info(f"Current progress: Checked {self.counter['total_checked']} projects, "
                                  f"Cloned {self.counter['total_cloned']} projects successfully")

                page_info = search_data.get('pageInfo', {})
                if not page_info.get('hasNextPage'):
                    break

                cursor = page_info.get('endCursor')
                page += 1
                pbar.update(1)
                
                self.random_delay()

        self.save_counter()
        
        logger.info(f"Crawling completed:")
        logger.info(f"- Total checked projects: {self.counter['total_checked']}")
        logger.info(f"- Total cloned projects (Python files {min_python_files}-{max_python_files}): {self.counter['total_cloned']}")
        logger.info(f"- Data saved to {self.projects_file}")
        logger.info(f"- Counter last updated: {self.counter['last_updated']}")

if __name__ == "__main__":
    try:
        crawler = GitHubCrawler()
        crawler.crawl()
    except Exception as e:
        logger.error(f"Program execution error: {e}")
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        crawler.save_counter()
        crawler.save_projects()
        logger.info(f"Current progress saved: Checked {crawler.counter['total_checked']} projects, "
                   f"Cloned {crawler.counter['total_cloned']} projects successfully")

