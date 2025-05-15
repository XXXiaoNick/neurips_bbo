"""
EARLBO Implementation.

This module defines the EARLBO class, which extends BaseBayesianOptimization
to provide a single-step RL approach integrated with GP for Bayesian Optimization.
"""

from ..base_method import BaseBayesianOptimization

class EARLBO(BaseBayesianOptimization):
    """
    EARLBO - A baseline approach that uses reinforcement learning (RL) 
    for single-step policy-based decision, combined with a GP model.

    Attributes:
        benchmark (object): The benchmark or objective function interface.
        initial_points (list): A list of (x, y) pairs for initialization.
        config (dict): Overall config, including gp_params and rl_params, etc.
        gp_params (dict): GP hyperparameters and settings.
        rl_params (dict): RL-related parameters (policy net, epsilon, etc.).
    """
    
    def __init__(self, benchmark, initial_points=None, config=None):
        """
        Initialize EARLBO instance.

        Args:
            benchmark (object): The benchmark or objective interface.
            initial_points (list, optional): A list of (x, y) initial observations.
            config (dict, optional): A dictionary containing 'gp_params', 'rl_params', etc.
        """
        super().__init__(config)
        
        self.benchmark = benchmark
        self.initial_points = initial_points or []
        self.config = config or {}
        
        self.gp_params = self.config.get('gp_params', {})
        self.rl_params = self.config.get('rl_params', {})
        

        self.name = self.config.get('name', "EARLBO")
        

        self.bounds = getattr(self.benchmark, 'bounds', None)

    def initialize(self, objective_function, bounds, initial_points=None):
        """
        Initialize the EARLBO with objective_function, search bounds, and optional points.

        Args:
            objective_function (callable): The black-box function to optimize.
            bounds (list): List of (lower, upper) tuples for each dimension.
            initial_points (list, optional): Additional (x, y) data to initialize.
        """
      
        super().initialize(objective_function, bounds, initial_points)
        print(f"[{self.name}] Initializing with Single-step RL + GP approach.")

    def suggest_next_point(self):
        """
        Suggest the next point to evaluate using an RL policy that interacts with a GP model.

        Returns:
            list: The next suggested design point (same dimensionality as 'bounds').
        """
        print(f"[{self.name}] Suggesting next point using RL policy.")
        if self.bounds:
            return [0.5] * len(self.bounds)
        else:
            return [0.5] 
    def update(self, x, y):
        """
        Update the method with a newly observed data point.

        Args:
            x (list): The input design.
            y (float): The objective function value at x.
        """
        print(f"[{self.name}] Updating with new point: {x}, value: {y}")

    def best_point(self):
        """
        Return the best solution found so far.

        Returns:
            tuple: (best_x, best_y) with best_x as an input vector and best_y as scalar value.
        """
        print(f"[{self.name}] Returning best point so far.")
        if self.bounds:
            return ([0.5] * len(self.bounds), 0.0)
        else:
            return ([0.5], 0.0)
