class BaseBayesianOptimization:
    def __init__(self, config):
        self.config = config
        self.name = config.get('name', 'BaseBO')
        
    def initialize(self, objective_function, bounds, initial_points=None):
        self.objective_function = objective_function
        self.bounds = bounds
        self.initial_points = initial_points
        
    def suggest_next_point(self):
        raise NotImplementedError("Subclasses must implement suggest_next_point()")
        
    def update(self, x, y):
        raise NotImplementedError("Subclasses must implement update()")
        
    def best_point(self):
        raise NotImplementedError("Subclasses must implement best_point()")


class BaseREBMBO(BaseBayesianOptimization):
    def __init__(self, config):
        super().__init__(config)
        self.variant = config.get('variant', 'base')
        
    def train_ebm(self):
        raise NotImplementedError("REBMBO subclasses must implement train_ebm()")
        
    def update_policy(self):
        raise NotImplementedError("REBMBO subclasses must implement update_policy()")
