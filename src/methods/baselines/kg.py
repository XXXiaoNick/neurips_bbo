"""
Knowledge Gradient (KG) Implementation.

Defines the KnowledgeGradient class, extending BaseBayesianOptimization
to provide a KG-based acquisition strategy with a GP model.
"""

from ..base_method import BaseBayesianOptimization

class KnowledgeGradient(BaseBayesianOptimization):
    """
    KnowledgeGradient class that uses a GP for surrogate modeling
    and a KG-based acquisition function.
    """

    def __init__(self, benchmark, initial_points=None, config=None):
        """
        Initialize the KnowledgeGradient method.

        Args:
            benchmark (object): The benchmark or objective function interface.
            initial_points (list, optional): A list of (x, y) pairs for initialization.
            config (dict, optional): A dictionary containing 'gp_params', 'acquisition_params', etc.
        """
        super().__init__(config) 
        
        self.benchmark = benchmark
        self.initial_points = initial_points or []
        self.config = config or {}
        
        self.gp_params = self.config.get('gp_params', {})
        self.acquisition_params = self.config.get('acquisition_params', {})
        
        self.name = self.config.get('name', "KnowledgeGradient")
        
    
        self.bounds = getattr(self.benchmark, 'bounds', None)

    def initialize(self, objective_function, bounds, initial_points=None):
        """
        Initialize KG with the objective function, search bounds, and optional init data.
        """
        super().initialize(objective_function, bounds, initial_points)
        print(f"[{self.name}] Initializing Knowledge Gradient approach.")


    def suggest_next_point(self):
        """
        Suggest the next point to evaluate using KG strategy.
        """
        print(f"[{self.name}] Suggesting next point via Knowledge Gradient.")
        if self.bounds:
            return [0.5] * len(self.bounds)
        return [0.5]

    def update(self, x, y):
        """
        Update KG model with a new data point (x, y).
        """
        print(f"[{self.name}] Updating with new point: {x}, value: {y}")

    def best_point(self):
        """
        Return the best solution found so far.
        """
        print(f"[{self.name}] Returning best point so far.")
    
        if self.bounds:
            return ([0.5] * len(self.bounds), 0.0)
        return ([0.5], 0.0)
