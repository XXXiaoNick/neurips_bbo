"""
Base Optimizer Interface Module.

This module defines the abstract Optimizer class that serves as the foundation
for all optimization methods in the project. It provides common functionality
for suggesting points, handling evaluations, and tracking optimization progress.
"""

import numpy as np
import time
import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Optimizer")

class Optimizer(ABC):
    """
    Abstract base class for all optimization methods.
    
    This class defines the interface that all optimizer implementations must follow,
    providing common utilities for suggesting points, handling evaluations, and
    tracking optimization progress.
    """
    
    def __init__(self, 
                 bounds: List[List[float]],
                 objective_func: Callable[[np.ndarray], float],
                 initial_points: Optional[np.ndarray] = None,
                 maximize: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the optimizer.
        
        Args:
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            objective_func (Callable): Function to optimize
            initial_points (np.ndarray, optional): Initial points to evaluate
            maximize (bool): Whether to maximize (True) or minimize (False) the objective
            name (str, optional): Name of the optimizer
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.objective_func = objective_func
        self.initial_points = initial_points
        self.maximize = maximize
        self.name = name or self.__class__.__name__
        
        # Internal state
        self.X_train = []  # Training inputs
        self.y_train = []  # Training outputs
        self.n_calls = 0   # Number of function evaluations
        
        # Tracking variables
        self.history = {
            'x': [],           # All evaluated points
            'y': [],           # All function values
            'best_x': [],      # Best point at each iteration
            'best_y': [],      # Best value at each iteration
            'time': [],        # Time spent on each iteration
            'memory': [],      # Memory usage for each iteration
            'model_time': [],  # Time spent on model updating
        }
        
        # Best observed values
        self.best_x = None
        self.best_y = float('-inf') if maximize else float('inf')
    
    @abstractmethod
    def suggest(self) -> np.ndarray:
        """
        Suggest next point to evaluate.
        
        Returns:
            np.ndarray: The suggested point
        """
        raise NotImplementedError("Subclasses must implement suggest method")
    
    def observe(self, x: np.ndarray, y: float) -> None:
        """
        Update model with new observation.
        
        Args:
            x (np.ndarray): The point that was evaluated
            y (float): The observed value at x
        """
        # Convert to numpy arrays if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Update training data
        self.X_train.append(x.copy())
        self.y_train.append(float(y))
        
        # Update best values (handle maximization/minimization)
        y_opt = y if self.maximize else -y
        if self.best_x is None or y_opt > (self.best_y if self.maximize else -self.best_y):
            self.best_x = x.copy()
            self.best_y = y
        
        # Update history
        self.history['x'].append(x.copy())
        self.history['y'].append(float(y))
        self.history['best_x'].append(self.best_x.copy())
        self.history['best_y'].append(self.best_y)
        
        # Call internal update method (to be implemented by subclasses)
        self._update(x, y)
    
    def _update(self, x: np.ndarray, y: float) -> None:
        """
        Update internal models with new observation (to be implemented by subclasses).
        
        Args:
            x (np.ndarray): The point that was evaluated
            y (float): The observed value at x
        """
        pass  # Default implementation does nothing
    
    def optimize(self, max_iter: int, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Run optimization loop.
        
        Args:
            max_iter (int): Maximum number of iterations
            verbose (bool): Whether to print progress
            
        Returns:
            Tuple[np.ndarray, float]: Best point found and its value
        """
        # Initialize with initial points if provided
        if self.initial_points is not None:
            for x_init in self.initial_points:
                start_time = time.time()
                
                # Evaluate the objective
                y = self.objective_func(x_init)
                self.n_calls += 1
                
                # Track iteration time
                iter_time = time.time() - start_time
                self.history['time'].append(iter_time)
                
                # Update the model
                model_start_time = time.time()
                self.observe(x_init, y)
                model_time = time.time() - model_start_time
                self.history['model_time'].append(model_time)
                
                if verbose:
                    self._print_progress(0, max_iter, x_init, y, iter_time)
        
        # Main optimization loop
        for i in range(max_iter):
            start_time = time.time()
            
            # Get suggestion
            try:
                x_next = self.suggest()
            except Exception as e:
                logger.error(f"Error suggesting point: {e}")
                break
            
            # Evaluate objective
            try:
                y_next = self.objective_func(x_next)
                self.n_calls += 1
            except Exception as e:
                logger.error(f"Error evaluating objective: {e}")
                continue
            
            # Track iteration time
            iter_time = time.time() - start_time
            self.history['time'].append(iter_time)
            
            # Update model
            model_start_time = time.time()
            self.observe(x_next, y_next)
            model_time = time.time() - model_start_time
            self.history['model_time'].append(model_time)
            
            # Print progress
            if verbose and ((i+1) % 1 == 0 or i == max_iter - 1):
                self._print_progress(i+1, max_iter, x_next, y_next, iter_time)
        
        return self.get_best()
    
    def _print_progress(self, iteration: int, max_iter: int, x: np.ndarray, y: float, iter_time: float) -> None:
        """
        Print optimization progress.
        
        Args:
            iteration (int): Current iteration
            max_iter (int): Maximum iterations
            x (np.ndarray): Current point
            y (float): Current value
            iter_time (float): Iteration time
        """
        x_str = np.array2string(x, precision=4, suppress_small=True)
        objective_type = "Maximizing" if self.maximize else "Minimizing"
        
        print(f"Iteration {iteration}/{max_iter} - "
              f"{objective_type} objective - "
              f"Point: {x_str} - "
              f"Value: {y:.6f} - "
              f"Best: {self.best_y:.6f} - "
              f"Time: {iter_time:.3f}s")
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        Get the best point found and its value.
        
        Returns:
            Tuple[np.ndarray, float]: Best point and its value
        """
        if self.best_x is None:
            return None, None
        return self.best_x.copy(), self.best_y
    
    def get_history(self) -> Dict[str, List]:
        """
        Get the optimization history.
        
        Returns:
            Dict[str, List]: The optimization history
        """
        return self.history
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self.X_train = []
        self.y_train = []
        self.n_calls = 0
        self.best_x = None
        self.best_y = float('-inf') if self.maximize else float('inf')
        
        # Clear history
        self.history = {
            'x': [],
            'y': [],
            'best_x': [],
            'best_y': [],
            'time': [],
            'memory': [],
            'model_time': [],
        }
    
    def save_state(self, filepath: str) -> None:
        """
        Save optimizer state to file.
        
        Args:
            filepath (str): Path to save the state
        """
        # Convert numpy arrays to lists for JSON serialization
        state = {
            'X_train': [x.tolist() for x in self.X_train],
            'y_train': self.y_train,
            'best_x': None if self.best_x is None else self.best_x.tolist(),
            'best_y': self.best_y,
            'n_calls': self.n_calls,
            'history': {
                'x': [x.tolist() for x in self.history['x']],
                'y': self.history['y'],
                'best_x': [x.tolist() for x in self.history['best_x']],
                'best_y': self.history['best_y'],
                'time': self.history['time'],
                'memory': self.history['memory'],
                'model_time': self.history['model_time'],
            },
            'metadata': {
                'dim': self.dim,
                'bounds': self.bounds.tolist(),
                'maximize': self.maximize,
                'name': self.name,
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Load optimizer state from file.
        
        Args:
            filepath (str): Path to the state file
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Convert lists back to numpy arrays
        self.X_train = [np.array(x) for x in state['X_train']]
        self.y_train = state['y_train']
        self.best_x = None if state['best_x'] is None else np.array(state['best_x'])
        self.best_y = state['best_y']
        self.n_calls = state['n_calls']
        
        # Load history
        self.history = {
            'x': [np.array(x) for x in state['history']['x']],
            'y': state['history']['y'],
            'best_x': [np.array(x) for x in state['history']['best_x']],
            'best_y': state['history']['best_y'],
            'time': state['history']['time'],
            'memory': state['history']['memory'],
            'model_time': state['history']['model_time'],
        }
        
        # Check metadata for compatibility
        metadata = state.get('metadata', {})
        if metadata.get('dim') != self.dim:
            logger.warning(f"Dimension mismatch: loaded state has dim={metadata.get('dim')}, but optimizer has dim={self.dim}")
        if metadata.get('maximize') != self.maximize:
            logger.warning(f"Objective type mismatch: loaded state has maximize={metadata.get('maximize')}, but optimizer has maximize={self.maximize}")
    
    def __str__(self) -> str:
        """String representation of the optimizer."""
        return (f"{self.name} (dim={self.dim}, "
                f"{'maximization' if self.maximize else 'minimization'}, "
                f"calls={self.n_calls})")
    
    def __repr__(self) -> str:
        """Detailed representation of the optimizer."""
        return (f"{self.__class__.__name__}(dim={self.dim}, "
                f"bounds={self.bounds.tolist()}, "
                f"maximize={self.maximize}, "
                f"name='{self.name}', "
                f"n_calls={self.n_calls})")


class RandomOptimizer(Optimizer):
    """
    Random search optimizer for benchmarking.
    
    This class implements a simple random search optimizer that samples points
    uniformly from the search space. It can be used as a baseline for comparison.
    """
    
    def __init__(self, 
                 bounds: List[List[float]],
                 objective_func: Callable[[np.ndarray], float],
                 initial_points: Optional[np.ndarray] = None,
                 maximize: bool = True,
                 name: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize the random optimizer.
        
        Args:
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            objective_func (Callable): Function to optimize
            initial_points (np.ndarray, optional): Initial points to evaluate
            maximize (bool): Whether to maximize (True) or minimize (False) the objective
            name (str, optional): Name of the optimizer
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__(bounds, objective_func, initial_points, maximize, name or "RandomSearch")
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
    
    def suggest(self) -> np.ndarray:
        """
        Suggest next point to evaluate.
        
        Returns:
            np.ndarray: The suggested point (sampled uniformly from bounds)
        """
        # Sample uniformly from bounds
        x = np.zeros(self.dim)
        for i in range(self.dim):
            x[i] = np.random.uniform(self.bounds[i, 0], self.bounds[i, 1])
        
        return x


class GridOptimizer(Optimizer):
    """
    Grid search optimizer for benchmarking.
    
    This class implements a simple grid search optimizer that systematically
    evaluates points on a grid defined by the specified resolution.
    """
    
    def __init__(self, 
                 bounds: List[List[float]],
                 objective_func: Callable[[np.ndarray], float],
                 initial_points: Optional[np.ndarray] = None,
                 maximize: bool = True,
                 name: Optional[str] = None,
                 resolution: int = 10):
        """
        Initialize the grid optimizer.
        
        Args:
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            objective_func (Callable): Function to optimize
            initial_points (np.ndarray, optional): Initial points to evaluate
            maximize (bool): Whether to maximize (True) or minimize (False) the objective
            name (str, optional): Name of the optimizer
            resolution (int): Resolution of the grid in each dimension
        """
        super().__init__(bounds, objective_func, initial_points, maximize, name or "GridSearch")
        
        self.resolution = resolution
        self._setup_grid()
    
    def _setup_grid(self) -> None:
        """Set up the evaluation grid."""
        # Create grid points for each dimension
        grid_points = []
        for i in range(self.dim):
            points = np.linspace(self.bounds[i, 0], self.bounds[i, 1], self.resolution)
            grid_points.append(points)
        
        # Create mesh grid
        mesh = np.meshgrid(*grid_points)
        
        # Reshape to get all points
        self.grid = np.vstack([m.flatten() for m in mesh]).T
        self.grid_index = 0
    
    def suggest(self) -> np.ndarray:
        """
        Suggest next point to evaluate.
        
        Returns:
            np.ndarray: The next point on the grid
        """
        if self.grid_index >= len(self.grid):
            # If we've exhausted the grid, return a random point
            logger.warning("Grid exhausted, returning random point")
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        # Get the next point on the grid
        x = self.grid[self.grid_index]
        self.grid_index += 1
        
        return x