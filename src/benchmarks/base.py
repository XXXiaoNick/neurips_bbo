"""
Base Benchmark Interface Module.

This module defines the abstract Benchmark class that serves as the foundation
for all benchmark implementations in the project. It provides common functionality
for objective function evaluation, tracking, and metrics calculation.
"""

import numpy as np
import time
import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union


class Benchmark(ABC):
    """
    Abstract base class for all benchmarks.
    
    This class defines the interface that all benchmark implementations must follow,
    providing common utilities for evaluating the objective function, handling bounds,
    tracking evaluations, and computing various metrics.
    """
    
    def __init__(self, 
                 dim: int,
                 bounds: List[List[float]],
                 noise_std: float = 0.0,
                 maximize: bool = True,
                 name: Optional[str] = None):
        """
        Initialize the benchmark.
        
        Args:
            dim (int): Dimensionality of the search space
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            noise_std (float): Standard deviation of observation noise
            maximize (bool): Whether the objective is to be maximized (True) or minimized (False)
            name (str, optional): Name of the benchmark
        """
        self.dim = dim
        self.bounds = np.array(bounds)
        
        # Handle case where bounds are provided as a single [min, max] pair for all dimensions
        if self.bounds.shape == (2,) or (len(self.bounds) == 2 and np.isscalar(self.bounds[0])):
            self.bounds = np.array([self.bounds] * dim)
            
        # Validate bounds
        if self.bounds.shape != (dim, 2):
            raise ValueError(f"Bounds should be of shape ({dim}, 2), got {self.bounds.shape}")
        
        self.noise_std = noise_std
        self.maximize = maximize
        self.name = name or self.__class__.__name__
        
        # Attributes for tracking evaluations
        self.n_calls = 0
        self.best_value = None
        self.true_optimum = None  # To be set by subclasses if known
        self.true_optimum_loc = None  # Location of the optimum if known
        
        # Evaluation history
        self.history = {
            "x": [],  # Inputs
            "y": [],  # Outputs
            "times": [],  # Evaluation times
            "best_y": [],  # Best value found so far
            "best_x": []   # Best point found so far
        }
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at point x.
        
        Args:
            x (np.ndarray): The point at which to evaluate the objective
            
        Returns:
            float: The objective value
        """
        # Convert to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        # Reshape if necessary
        if x.ndim == 0:  # Scalar case
            x = np.array([x])
        elif x.ndim > 1:  # Excess dimensions
            x = x.flatten()
            
        # Validate dimensions
        if len(x) != self.dim:
            raise ValueError(f"Expected input of dimension {self.dim}, got {len(x)}")
        
        # Clip to bounds
        x_clipped = np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
        if not np.allclose(x, x_clipped):
            print(f"Warning: Input {x} outside bounds, clipping to {x_clipped}")
            x = x_clipped
        
        # Increment call counter
        self.n_calls += 1
        
        # Start timer
        start_time = time.time()
        
        # Evaluate function
        y = self._evaluate(x)
        
        # Apply noise if specified
        if self.noise_std > 0:
            y += np.random.normal(0, self.noise_std)
        
        # Flip sign if minimizing
        y_orig = y
        if not self.maximize:
            y = -y
        
        # End timer
        evaluation_time = time.time() - start_time
        
        # Update best value
        if self.best_value is None or y > self.best_value:
            self.best_value = y
            best_x = x.copy()
        else:
            best_x = self.history["best_x"][-1] if self.history["best_x"] else x.copy()
        
        # Update history
        self.history["x"].append(x.copy())
        self.history["y"].append(y_orig)  # Store original value
        self.history["times"].append(evaluation_time)
        self.history["best_y"].append(self.best_value if self.maximize else -self.best_value)
        self.history["best_x"].append(best_x)
        
        return y_orig if self.maximize else y
    
    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at point x (to be implemented by subclasses).
        
        Args:
            x (np.ndarray): The point at which to evaluate the objective
            
        Returns:
            float: The objective value
        """
        raise NotImplementedError("Subclasses must implement _evaluate method")
    
    def get_optimal_value(self) -> Optional[float]:
        """
        Get the true optimal value if known.
        
        Returns:
            float or None: The optimal value or None if not known
        """
        return self.true_optimum
    
    def get_optimal_point(self) -> Optional[np.ndarray]:
        """
        Get the true optimal point if known.
        
        Returns:
            np.ndarray or None: The optimal point or None if not known
        """
        return self.true_optimum_loc
    
    def get_pseudo_regret(self, value: float) -> Optional[float]:
        """
        Calculate the pseudo-regret for a given value.
        
        Args:
            value (float): The value for which to calculate regret
            
        Returns:
            float or None: The pseudo-regret or None if optimal value is not known
        """
        if self.true_optimum is None:
            return None
        
        # For maximization: regret = optimal - value
        # For minimization: regret = value - optimal
        if self.maximize:
            return abs(self.true_optimum - value)
        else:
            return abs(value - self.true_optimum)
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """
        Get the best point found and its value.
        
        Returns:
            Tuple[np.ndarray, float]: The best point and its value
        """
        if not self.history["best_x"]:
            return None, None
        
        best_x = self.history["best_x"][-1]
        best_y = self.history["best_y"][-1]
        
        return best_x, best_y
    
    def get_history(self) -> Dict[str, List]:
        """
        Get the evaluation history.
        
        Returns:
            Dict[str, List]: The evaluation history
        """
        return self.history
    
    def reset(self) -> None:
        """Reset the benchmark state."""
        self.n_calls = 0
        self.best_value = None
        self.history = {
            "x": [],
            "y": [],
            "times": [],
            "best_y": [],
            "best_x": []
        }
    
    def save_history(self, filepath: str) -> None:
        """
        Save evaluation history to file.
        
        Args:
            filepath (str): Path to save the history
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {
            "x": [x.tolist() for x in self.history["x"]],
            "y": self.history["y"],
            "times": self.history["times"],
            "best_y": self.history["best_y"],
            "best_x": [x.tolist() for x in self.history["best_x"]],
            "metadata": {
                "dim": self.dim,
                "bounds": self.bounds.tolist(),
                "noise_std": self.noise_std,
                "maximize": self.maximize,
                "name": self.name,
                "n_calls": self.n_calls,
                "true_optimum": self.true_optimum,
                "true_optimum_loc": None if self.true_optimum_loc is None 
                                    else self.true_optimum_loc.tolist()
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    @classmethod
    def load_history(cls, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation history from file.
        
        Args:
            filepath (str): Path to the history file
            
        Returns:
            Dict[str, Any]: The loaded history
        """
        with open(filepath, 'r') as f:
            history = json.load(f)
        
        # Convert lists back to numpy arrays
        history["x"] = [np.array(x) for x in history["x"]]
        history["best_x"] = [np.array(x) for x in history["best_x"]]
        
        # Extract metadata
        metadata = history.pop("metadata", {})
        
        return history, metadata
    
    def __str__(self) -> str:
        """String representation of the benchmark."""
        return (f"{self.name} (dim={self.dim}, "
                f"{'maximization' if self.maximize else 'minimization'}, "
                f"noise_std={self.noise_std})")
    
    def __repr__(self) -> str:
        """Detailed representation of the benchmark."""
        return (f"{self.__class__.__name__}(dim={self.dim}, "
                f"bounds={self.bounds.tolist()}, "
                f"noise_std={self.noise_std}, "
                f"maximize={self.maximize}, "
                f"name='{self.name}')")


class SyntheticBenchmark(Benchmark):
    """
    Base class for synthetic benchmarks with known optima.
    
    This class extends the Benchmark class for synthetic functions where
    the true optimum value and/or location are known in advance.
    """
    
    def __init__(self, 
                 dim: int,
                 bounds: List[List[float]],
                 noise_std: float = 0.0,
                 maximize: bool = True,
                 name: Optional[str] = None,
                 true_optimum: Optional[float] = None,
                 true_optimum_loc: Optional[np.ndarray] = None):
        """
        Initialize the synthetic benchmark.
        
        Args:
            dim (int): Dimensionality of the search space
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            noise_std (float): Standard deviation of observation noise
            maximize (bool): Whether the objective is to be maximized (True) or minimized (False)
            name (str, optional): Name of the benchmark
            true_optimum (float, optional): The true optimal value (if known)
            true_optimum_loc (np.ndarray, optional): The true optimal location (if known)
        """
        super().__init__(dim, bounds, noise_std, maximize, name)
        
        self.true_optimum = true_optimum
        self.true_optimum_loc = true_optimum_loc


class RealWorldBenchmark(Benchmark):
    """
    Base class for real-world benchmarks.
    
    This class extends the Benchmark class for real-world problems where
    evaluating the objective function may involve calling external simulators,
    APIs, or databases.
    """
    
    def __init__(self, 
                 dim: int,
                 bounds: List[List[float]],
                 noise_std: float = 0.0,
                 maximize: bool = True,
                 name: Optional[str] = None,
                 simulator_path: Optional[str] = None,
                 cache_results: bool = True):
        """
        Initialize the real-world benchmark.
        
        Args:
            dim (int): Dimensionality of the search space
            bounds (List[List[float]]): List of [min, max] bounds for each dimension
            noise_std (float): Standard deviation of observation noise
            maximize (bool): Whether the objective is to be maximized (True) or minimized (False)
            name (str, optional): Name of the benchmark
            simulator_path (str, optional): Path to the simulator or data file
            cache_results (bool): Whether to cache evaluation results to avoid recomputation
        """
        super().__init__(dim, bounds, noise_std, maximize, name)
        
        self.simulator_path = simulator_path
        self.cache_results = cache_results
        self.cache = {}  # For storing previously computed results
        
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at point x, with caching if enabled.
        
        Args:
            x (np.ndarray): The point at which to evaluate the objective
            
        Returns:
            float: The objective value
        """
        # Convert to numpy array if needed
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        # Check cache if enabled
        if self.cache_results:
            x_tuple = tuple(x.flatten())
            if x_tuple in self.cache:
                # Still track the call but don't recompute
                self.n_calls += 1
                y = self.cache[x_tuple]
                self.history["x"].append(x.copy())
                self.history["y"].append(y)
                self.history["times"].append(0.0)  # No computation time
                
                # Update best value if needed
                y_opt = y if self.maximize else -y
                if self.best_value is None or y_opt > self.best_value:
                    self.best_value = y_opt
                    best_x = x.copy()
                else:
                    best_x = self.history["best_x"][-1] if self.history["best_x"] else x.copy()
                
                self.history["best_y"].append(self.best_value if self.maximize else -self.best_value)
                self.history["best_x"].append(best_x)
                
                return y
        
        # Call the parent implementation for actual evaluation
        y = super().__call__(x)
        
        # Update cache if enabled
        if self.cache_results:
            self.cache[tuple(x.flatten())] = y
            
        return y
    
    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self.cache = {}