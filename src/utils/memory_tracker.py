"""
Memory and resource tracking utilities for optimization algorithms.
This module provides tools to monitor CPU/GPU memory usage and execution time.
"""

import time
import os
import psutil
import numpy as np
from typing import Callable, Any, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResourceTracker")

class ResourceTracker:
    """Tracks CPU/GPU memory usage and execution time during optimization."""
    
    def __init__(self, enable_gpu: bool = True, log_level: str = "INFO"):
        """
        Initialize the resource tracker.
        
        Args:
            enable_gpu (bool): Whether to track GPU memory (if available)
            log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        self.process = psutil.Process(os.getpid())
        self.enable_gpu = enable_gpu
        self.gpu_available = False
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Check for GPU availability
        if self.enable_gpu:
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
                if self.gpu_available:
                    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                    self.torch = torch
                else:
                    logger.info("No GPU detected. GPU tracking disabled.")
            except ImportError:
                logger.warning("PyTorch not installed. GPU tracking disabled.")
                self.gpu_available = False
    
    def get_cpu_memory(self) -> float:
        """
        Get current CPU memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        return self.process.memory_info().rss / (1024 * 1024)
    
    def get_gpu_memory(self) -> float:
        """
        Get current GPU memory usage in MB.
        
        Returns:
            float: Memory usage in MB, or 0 if GPU is not available
        """
        if self.gpu_available:
            try:
                return self.torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception as e:
                logger.error(f"Error getting GPU memory: {e}")
                return 0
        return 0
    
    def get_memory(self) -> Dict[str, float]:
        """
        Get both CPU and GPU memory usage.
        
        Returns:
            Dict[str, float]: Dictionary with 'cpu' and 'gpu' memory usage in MB
        """
        mem_info = {
            'cpu': self.get_cpu_memory(),
            'gpu': self.get_gpu_memory() if self.gpu_available else 0
        }
        return mem_info
    
    def time_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, float, Dict[str, float]]:
        """
        Time a function execution and track memory usage.
        
        Args:
            func (Callable): Function to time
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple[Any, float, Dict[str, float]]: 
                - Function result
                - Execution time in seconds
                - Dictionary with memory usage before and after execution
        """
        # Get memory before
        mem_before = self.get_memory()
        
        # Time the function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get memory after
        mem_after = self.get_memory()
        
        # Calculate memory difference
        mem_diff = {
            'cpu': mem_after['cpu'] - mem_before['cpu'],
            'gpu': mem_after['gpu'] - mem_before['gpu'] if self.gpu_available else 0
        }
        
        execution_time = end_time - start_time
        logger.debug(f"Function executed in {execution_time:.4f} seconds. "
                    f"Memory change: CPU {mem_diff['cpu']:.2f} MB, "
                    f"GPU {mem_diff['gpu']:.2f} MB")
        
        return result, execution_time, mem_diff
    
    def log_resources(self, tag: str = "") -> Dict[str, float]:
        """
        Log current resource usage.
        
        Args:
            tag (str): Optional tag for the log message
            
        Returns:
            Dict[str, float]: Dictionary with current resource usage
        """
        mem = self.get_memory()
        tag_prefix = f"[{tag}] " if tag else ""
        logger.info(f"{tag_prefix}Memory usage: CPU {mem['cpu']:.2f} MB, GPU {mem['gpu']:.2f} MB")
        return mem


class ResourceHistory:
    """Keeps track of resource usage history during optimization."""
    
    def __init__(self):
        """Initialize the resource history tracker."""
        self.timestamps = []
        self.cpu_memory = []
        self.gpu_memory = []
        self.iteration_times = []
        
    def record(self, timestamp: float, cpu_mem: float, gpu_mem: float, iteration_time: Optional[float] = None):
        """
        Record resource usage at a point in time.
        
        Args:
            timestamp (float): Timestamp in seconds
            cpu_mem (float): CPU memory usage in MB
            gpu_mem (float): GPU memory usage in MB
            iteration_time (float, optional): Time taken for the current iteration
        """
        self.timestamps.append(timestamp)
        self.cpu_memory.append(cpu_mem)
        self.gpu_memory.append(gpu_mem)
        if iteration_time is not None:
            self.iteration_times.append(iteration_time)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate resource usage statistics.
        
        Returns:
            Dict: Statistics including mean, max, min, std for CPU/GPU memory and iteration time
        """
        stats = {}
        
        # Function to calculate stats for a metric
        def calc_stats(data):
            if not data:
                return {
                    'mean': 0.0,
                    'max': 0.0,
                    'min': 0.0,
                    'std': 0.0,
                    'total': 0.0
                }
            data_array = np.array(data)
            return {
                'mean': float(np.mean(data_array)),
                'max': float(np.max(data_array)),
                'min': float(np.min(data_array)),
                'std': float(np.std(data_array)),
                'total': float(np.sum(data_array))
            }
        
        stats['cpu_memory'] = calc_stats(self.cpu_memory)
        stats['gpu_memory'] = calc_stats(self.gpu_memory)
        stats['iteration_time'] = calc_stats(self.iteration_times)
        
        return stats
    
    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage.
        
        Returns:
            Dict[str, float]: Peak CPU and GPU memory usage in MB
        """
        peak = {
            'cpu': max(self.cpu_memory) if self.cpu_memory else 0,
            'gpu': max(self.gpu_memory) if self.gpu_memory else 0
        }
        return peak
    
    def get_total_time(self) -> float:
        """
        Get total execution time.
        
        Returns:
            float: Total execution time in seconds
        """
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]
    
    def to_dict(self) -> Dict[str, list]:
        """
        Convert history to dictionary for serialization.
        
        Returns:
            Dict[str, list]: Dictionary representation of the history
        """
        return {
            'timestamps': self.timestamps,
            'cpu_memory': self.cpu_memory,
            'gpu_memory': self.gpu_memory,
            'iteration_times': self.iteration_times
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> 'ResourceHistory':
        """
        Create ResourceHistory from dictionary.
        
        Args:
            data (Dict[str, list]): Dictionary representation of the history
            
        Returns:
            ResourceHistory: New instance with loaded data
        """
        history = cls()
        history.timestamps = data.get('timestamps', [])
        history.cpu_memory = data.get('cpu_memory', [])
        history.gpu_memory = data.get('gpu_memory', [])
        history.iteration_times = data.get('iteration_times', [])
        return history


# Convenience function to create a global tracker
def create_tracker(enable_gpu=True, log_level="INFO") -> ResourceTracker:
    """
    Create a global resource tracker instance.
    
    Args:
        enable_gpu (bool): Whether to track GPU memory
        log_level (str): Logging level
        
    Returns:
        ResourceTracker: Resource tracker instance
    """
    return ResourceTracker(enable_gpu=enable_gpu, log_level=log_level)