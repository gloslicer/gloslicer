import os
import torch
import time
import traceback

def log_gpu_memory(log_file=None, prefix=""):
    """
    Log current GPU memory usage
    
    Args:
        log_file: Path to log file, if None only prints to console
        prefix: Log prefix to distinguish different GPU recording scenarios
    """
    def log_message(msg):
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    try:
        if torch.cuda.is_available():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_message(f"\n--- {prefix} GPU Memory Usage ({timestamp}) ---")
            
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                cached = torch.cuda.memory_reserved(i) / (1024 ** 3)
                log_message(f"CUDA:{i} Allocated Memory: {allocated:.2f} GB, Cached Memory: {cached:.2f} GB")
            log_message("-----------------------------------")
    except Exception as e:
        log_message(f"GPU Memory Logging Error: {str(e)}")

def gpu_monitor(stop_event, log_file=None):
    """
    Monitor GPU usage every 10 seconds
    
    Args:
        stop_event: Event object to control monitoring thread stop
        log_file: Path to log file, if None only prints to console
    """
    log_message = lambda msg: print(msg) if not log_file else (print(msg), open(log_file, 'a').write(msg + '\n'))
    
    log_message("\n=== Starting GPU Monitoring ===")
    while not stop_event.is_set():
        try:
            log_gpu_memory(log_file, "GPU Monitor")
        except Exception as e:
            log_message(f"GPU Monitoring Error: {str(e)}")
        
        # Wait for 10 seconds
        for _ in range(10):
            if stop_event.is_set():
                break
            time.sleep(1)
    
    log_message("=== GPU Monitoring Ended ===")

def set_seed(seed):
    """
    Set random seed to ensure experiment reproducibility
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True 