"""
Synthetic benchmark functions implementation.

This module implements various synthetic benchmark functions commonly used
for testing optimization algorithms, such as Branin, Rosenbrock, etc.
"""

import numpy as np
from .base import Benchmark

class BraninBenchmark(Benchmark):
    """
    Branin function (2D)
    
    A standard low-dimensional test function with three global minima,
    commonly used for testing BO algorithm performance.
    
    Formula:
    f(x1,x2) = (x2 - 5.1/(4*pi^2)*x1^2 + 5/pi*x1 - 6)^2 + 10*(1-1/(8*pi))*cos(x1) + 10
    """
    def __init__(self, **kwargs):
        # Process parameters that may be passed from the configuration file
        dim = kwargs.get('dim', 2)
        bounds = kwargs.get('bounds', [(-5, 10), (0, 15)])
        name = kwargs.get('name', "Branin")
        maximize = kwargs.get('maximize', False)  # Default is minimization
        noise_std = kwargs.get('noise_std', 0.0)
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            noise_std=noise_std,
            maximize=maximize,
            name=name
        )
        
        # Store true optimum information
        self.true_optimum = kwargs.get('true_optimum', 0.397887)
        
        # Branin function has three global minima
        default_opt_locs = [
            (-np.pi, 12.275),
            (np.pi, 2.275),
            (9.42478, 2.475)
        ]
        self.true_optimum_loc = kwargs.get('true_optimum_loc', default_opt_locs[0])
    
    def _evaluate(self, x):
        """
        Calculate the Branin function value (abstract method implementation)
        
        Args:
            x: Input point, numpy array shape=(2,)
            
        Returns:
            Function value (float)
        """
        x1, x2 = x
        
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s
        
        # Note: The base class will automatically handle the sign based on the maximize parameter
        return term1 + term2 + term3


class RosenbrockBenchmark(Benchmark):
    """
    Rosenbrock function (variable dimensions, default 5D)
    
    A classic non-convex function with a global minimum located in a narrow, 
    curved valley, widely used to test an optimization algorithm's ability
    to escape local optima.
    
    Formula:
    f(x) = sum_{i=1}^{d-1} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
    """
    def __init__(self, **kwargs):
        # Process parameters that may be passed from the configuration file
        dim = kwargs.get('dim', 5)
        bounds = kwargs.get('bounds', [(-5, 10)] * dim)
        name = kwargs.get('name', f"Rosenbrock{dim}D")
        maximize = kwargs.get('maximize', False)  # Default is minimization
        noise_std = kwargs.get('noise_std', 0.0)
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            noise_std=noise_std,
            maximize=maximize,
            name=name
        )
        
        # Store true optimum information
        self.true_optimum = kwargs.get('true_optimum', 0.0)
        self.true_optimum_loc = kwargs.get('true_optimum_loc', np.ones(dim))
    
    def _evaluate(self, x):
        """
        Calculate the Rosenbrock function value (abstract method implementation)
        
        Args:
            x: Input point, numpy array shape=(dim,)
            
        Returns:
            Function value (float)
        """
        result = 0
        for i in range(self.dim - 1):
            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            
        return result


class AckleyBenchmark(Benchmark):
    """
    Ackley function (variable dimensions, default 5D)
    
    A multimodal function with many local minima and one global minimum,
    used to test the global search capability of optimization algorithms.
    
    Formula:
    f(x) = -20*exp(-0.2*sqrt(1/d*sum(x_i^2))) - exp(1/d*sum(cos(2*pi*x_i))) + 20 + e
    """
    def __init__(self, **kwargs):
        # Process parameters that may be passed from the configuration file
        dim = kwargs.get('dim', 5)
        bounds = kwargs.get('bounds', [(-32.768, 32.768)] * dim)
        name = kwargs.get('name', f"Ackley{dim}D")
        maximize = kwargs.get('maximize', False)  # Default is minimization
        noise_std = kwargs.get('noise_std', 0.0)
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            noise_std=noise_std,
            maximize=maximize,
            name=name
        )
        
        # Store true optimum information
        self.true_optimum = kwargs.get('true_optimum', 0.0)
        self.true_optimum_loc = kwargs.get('true_optimum_loc', np.zeros(dim))
    
    def _evaluate(self, x):
        """
        Calculate the Ackley function value (abstract method implementation)
        
        Args:
            x: Input point, numpy array shape=(dim,)
            
        Returns:
            Function value (float)
        """
        d = self.dim
        sum1 = np.sum(np.square(x))
        sum2 = np.sum(np.cos(2 * np.pi * x))
        
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        
        return term1 + term2 + 20 + np.e


class HDEXPBenchmark(Benchmark):
    """
    High-dimensional exponential sum function (variable dimensions, default 200D)
    
    A high-dimensional function designed to test the scalability of algorithms
    in high-dimensional spaces.
    
    Formula:
    f(x) = sum_{i=1}^{d} exp(x_i)
    """
    def __init__(self, **kwargs):
        # Process parameters that may be passed from the configuration file
        dim = kwargs.get('dim', 200)
        
        # Fix bounds handling for high-dimensional case
        if 'bounds' in kwargs:
            bounds_input = kwargs['bounds']
            if len(bounds_input) == 1:
                # If only one bound is provided, repeat it for all dimensions
                bounds = bounds_input * dim
            else:
                bounds = bounds_input
        else:
            # Default bounds
            bounds = [(-5, 5)] * dim
            
        name = kwargs.get('name', "HDBO")
        maximize = kwargs.get('maximize', True)  # Default is maximization
        noise_std = kwargs.get('noise_std', 0.0)
        
        super().__init__(
            dim=dim,
            bounds=bounds,
            noise_std=noise_std,
            maximize=maximize,
            name=name
        )
        
        # Calculate true optimum value (value at upper bound for each dimension)
        opt_x = np.ones(dim) * 5
        self.true_optimum = kwargs.get('true_optimum', np.sum(np.exp(opt_x)))
        self.true_optimum_loc = kwargs.get('true_optimum_loc', opt_x)
    
    def _evaluate(self, x):
        """
        Calculate the high-dimensional exponential sum function value (abstract method implementation)
        
        Args:
            x: Input point, numpy array shape=(dim,)
            
        Returns:
            Function value (float)
        """
        return np.sum(np.exp(x))

# Noisy variants

class BraninNoisyBenchmark(BraninBenchmark):
    """Noisy variant of the Branin function with added Gaussian noise"""
    
    def __init__(self, **kwargs):
        # Set default noise level but allow override through kwargs
        kwargs['noise_std'] = kwargs.get('noise_std', 0.1)
        kwargs['name'] = kwargs.get('name', "BraninNoisy")
        super().__init__(**kwargs)


class AckleyNoisyBenchmark(AckleyBenchmark):
    """Noisy variant of the Ackley function with added Gaussian noise"""
    
    def __init__(self, **kwargs):
        # Set default noise level but allow override through kwargs
        kwargs['noise_std'] = kwargs.get('noise_std', 0.2)
        kwargs['name'] = kwargs.get('name', "AckleyNoisy")
        super().__init__(**kwargs)