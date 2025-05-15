"""
Experiment Running Framework (experiment.py)

A fixed version that supports the structure where config['benchmarks'] is 
{ category: [ { name, class, params }, ... ] }, ensuring correct indentation
and avoiding syntax errors or missing blocks after else statements.

Usage:
    python experiment.py --config ./configs/experiments.yaml [--run] [--analyze]
"""

import os
import time
import numpy as np
import pickle
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import psutil
from tqdm import tqdm
import importlib
import argparse

# Optional: If your project has these base classes,
# handle if they are missing:
try:
    from benchmarks.base import Benchmark
    from methods.base import Optimizer, RandomOptimizer
except ImportError:
    print("Warning: Could not import base classes from benchmarks/base or methods/base. Some functionality may be limited.")
    Benchmark = object
    Optimizer = object
    RandomOptimizer = None

# If you have a registry in methods, import or define a fallback:
def get_method_class(method_name: str):
    """
    Dummy/fallback function for retrieving a method class by name.
    If your code actually has a registry, replace with the real logic.
    """
    return None  # For demonstration; real version might query a dictionary.

class Experiment:
    def __init__(self, config_path: str):
        """Initialize the experiment, parse config file and create output directories."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Result output directories
        general_cfg = self.config.get('general', {})
        self.results_dir = Path(general_cfg.get('output_dir', './results'))
        self.raw_dir = self.results_dir / 'raw'
        self.figures_dir = self.results_dir / 'figures'
        self.tables_dir = self.results_dir / 'tables'
        
        # Create result directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize empty dictionaries
        self.benchmarks = {}
        self.methods = {}
        
        # Parse benchmarks
        if 'benchmarks' in self.config:
            # structure: { "synthetic": [ { name, class, params }, ... ], "real_world": [ ... ] }
            for category_name, bench_list in self.config['benchmarks'].items():
                # bench_list should be a list
                if not isinstance(bench_list, list):
                    # In case the configuration is not a list but a dict
                    raise ValueError(f"Expected a list under benchmarks['{category_name}'], but got {type(bench_list)}.")
                
                for bench_cfg in bench_list:
                    # bench_cfg is { "name": "...", "class": "...", "params": {...} }
                    bench_name = bench_cfg['name']
                    bench_class_path = bench_cfg['class']
                    bench_params = bench_cfg.get('params', {})
                    
                    try:
                        bench_class = self._import_class(bench_class_path)
                        # Create benchmark instance
                        benchmark_instance = bench_class(**bench_params)
                        # Store in our dictionary
                        self.benchmarks[bench_name] = benchmark_instance
                    except Exception as e:
                        print(f"Error loading benchmark {bench_name}: {e}")
        
        # Parse methods
        if 'methods' in self.config:
            # structure: e.g. { "rebmbo": [...], "baselines": [...] }
            for category_name, method_list in self.config['methods'].items():
                if not isinstance(method_list, list):
                    raise ValueError(f"Expected a list under methods['{category_name}'], but got {type(method_list)}.")
                
                for m_cfg in method_list:
                    # Get method name
                    method_name = m_cfg['name']
                    
                    if 'class' in m_cfg:
                        # A direct class path is specified
                        method_class_path = m_cfg['class']
                        method_params = m_cfg.get('params', {})
                        try:
                            method_class = self._import_class(method_class_path)
                        except (ImportError, AttributeError) as e:
                            print(f"Warning: Failed to import {method_class_path} for method {method_name}: {e}")
                            print(f"Falling back to RandomOptimizer for {method_name}")
                            # fallback
                            if RandomOptimizer is not None:
                                method_class = RandomOptimizer
                            else:
                                method_class = self._create_dummy_optimizer()
                    else:
                        # If 'class' is missing, try registry or fallback
                        print(f"Warning: No 'class' specified for method {method_name}. Attempting registry fallback...")
                        method_class = get_method_class(method_name)
                        if method_class:
                            print(f"Created {method_name} from registry get_method_class")
                            method_params = m_cfg.get('params', {})
                            # save and continue
                            self.methods[method_name] = {
                                'class': method_class,
                                'config': method_params
                            }
                            continue
                        else:
                            print(f"Fallback to RandomOptimizer for {method_name}")
                            if RandomOptimizer is not None:
                                method_class = RandomOptimizer
                            else:
                                method_class = self._create_dummy_optimizer()
                            
                            # In that scenario we might just keep minimal params
                            method_params = m_cfg.get('params', {})
                            
                            # Optionally load config from file if specified
                            if 'config_file' in m_cfg:
                                config_file = m_cfg['config_file']
                                print(f"Attempting to load method config from {config_file} for {method_name}")
                                try:
                                    with open(config_file, 'r') as f:
                                        file_config = yaml.safe_load(f)
                                        # you might parse it further...
                                except Exception as e:
                                    print(f"Failed to load config from {config_file}: {e}")
                    
                    # Save the method
                    self.methods[method_name] = {
                        'class': method_class,
                        'config': method_params
                    }
        
        # Number of repetitions
        self.num_repetitions = int(self.config.get('general', {}).get('repetitions', 5))
    
    def _create_dummy_optimizer(self):
        """Create a basic optimizer class as fallback when imports fail."""
        class DummyOptimizer:
            def __init__(self, benchmark=None, initial_points=None, config=None):
                self.benchmark = benchmark
                self.initial_points = initial_points or []
                self.config = config or {}
                self.bounds = getattr(benchmark, 'bounds', [[0, 1]])
                self.dim = len(self.bounds)
            def suggest_next_point(self):
                x = np.zeros(self.dim)
                for i,(lo,hi) in enumerate(self.bounds):
                    x[i] = np.random.uniform(lo, hi)
                return x
            def update(self, x, y):
                pass
        return DummyOptimizer
    
    def _import_class(self, class_path: str):
        """
        Import a class from a string path, e.g. 'methods.rebmbo.classic.REBMBO_Classic'.
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except Exception as e:
            print(f"Error importing {class_path}: {e}")
            raise
    
    def run_all(self):
        """
        Run all benchmark and method combos. Return results = 
        { benchmark_name: { method_name: [run_res,...], ...}, ... }.
        Also saves intermediate results to self.raw_dir/results.pkl.
        """
        results = {}
        for b_name, b_obj in tqdm(self.benchmarks.items(), desc="Benchmarks"):
            results[b_name] = {}
            for m_name, m_info in tqdm(self.methods.items(), desc=f"Methods for {b_name}", leave=False):
                results[b_name][m_name] = []
                for rep in tqdm(range(self.num_repetitions), desc=f"{m_name} on {b_name}", leave=False):
                    res = self.run_single(b_name, b_obj, m_name, m_info, rep)
                    results[b_name][m_name].append(res)
                    self._save_raw_results(results)  # intermediate save
        return results
    
    def run_single(self, benchmark_name, benchmark_obj, method_name, method_info, rep):
        """
        Run a single experiment for a specific benchmark, method, and repetition index rep.
        """
        np.random.seed(rep)
        
        method_class = method_info['class']
        method_config = method_info['config']
        
        num_init = method_config.get('num_initial', 5)
        initial_points = []
        
        if not hasattr(benchmark_obj, 'bounds'):
            raise ValueError(f"Benchmark {benchmark_name} missing 'bounds' attribute.")
        
        # Generate initial points
        for _ in range(num_init):
            x = np.zeros(len(benchmark_obj.bounds))
            for i,(lo,hi) in enumerate(benchmark_obj.bounds):
                x[i] = np.random.uniform(lo, hi)
            try:
                y = benchmark_obj.__call__(x)
                initial_points.append((x,y))
            except Exception as e:
                print(f"Error evaluating {benchmark_name} at {x}: {e}")
                initial_points.append((x,0.0))
        
        try:
            # Instantiate the method
            method_instance = method_class(benchmark_obj, initial_points, method_config)
            
            # Determine budget
            budget = self._get_budget(benchmark_obj)
            
            start_time = time.time()
            regret_curve = []
            best_y = max(pt[1] for pt in initial_points)
            
            memory_usage = []
            times_each_iter = []
            
            # if there's a known optimum
            opt_y = None
            if hasattr(benchmark_obj, 'get_optimum'):
                try:
                    opt_data = benchmark_obj.get_optimum()
                    if opt_data and len(opt_data)==2:
                        opt_y = opt_data[1]
                except Exception as e:
                    print(f"Error calling get_optimum on {benchmark_name}: {e}")
            
            # main loop
            for i in range(budget - num_init):
                mem = self._get_memory_usage()
                memory_usage.append(mem)
                
                t0 = time.time()
                try:
                    next_x = method_instance.suggest_next_point()
                except Exception as e:
                    print(f"Error in {method_name}.suggest_next_point: {e}")
                    next_x = np.zeros(len(benchmark_obj.bounds))
                    for j,(lo,hi) in enumerate(benchmark_obj.bounds):
                        next_x[j] = np.random.uniform(lo,hi)
                
                try:
                    next_y = benchmark_obj.__call__(next_x)
                except Exception as e:
                    print(f"Error evaluating {benchmark_name} at {next_x}: {e}")
                    next_y = best_y
                
                dt = time.time()-t0
                times_each_iter.append(dt)
                
                try:
                    method_instance.update(next_x, next_y)
                except Exception as e:
                    print(f"Error updating {method_name} with new point: {e}")
                
                if next_y>best_y:
                    best_y = next_y
                
                # pseudo-regret
                if opt_y is not None:
                    r = opt_y - best_y
                else:
                    r = -best_y
                regret_curve.append(r)
            
            total_time = time.time()-start_time
            avg_mem = float(np.mean(memory_usage)) if memory_usage else 0.0
            avg_t = float(np.mean(times_each_iter)) if times_each_iter else 0.0
            
            # find convergence iteration
            conv_iter = None
            if regret_curve:
                final_r = regret_curve[-1]
                threshold = final_r * 1.05
                for idx,val in enumerate(regret_curve):
                    if val<=threshold:
                        conv_iter=idx
                        break
            
            return {
                "regret_curve": regret_curve,
                "best_y": best_y,
                "total_time": total_time,
                "avg_iter_time": avg_t,
                "avg_memory": avg_mem,
                "convergence_iter": conv_iter,
                "rep": rep
            }
        
        except Exception as e:
            print(f"Error in method {method_name} on benchmark {benchmark_name}, rep={rep}: {e}")
            return {
                "error": str(e),
                "regret_curve": regret_curve if 'regret_curve' in locals() else [],
                "best_y": best_y if 'best_y' in locals() else None
            }
    
    def analyze_results(self, results=None):
        """
        Perform analysis on results. If None, attempt to load from disk.
        """
        if results is None:
            results = self._load_raw_results()
        if not results:
            print("No results to analyze.")
            return None
        
        # TODO: table/figure code
        print("Analysis code not yet implemented.")
        return None
    
    def _save_raw_results(self, results):
        """Save results to pickle."""
        path = self.raw_dir / "results.pkl"
        try:
            with open(path, "wb") as f:
                pickle.dump(results, f)
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def _load_raw_results(self):
        """Load results from pickle if exists."""
        path = self.raw_dir / "results.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading results: {e}")
        return None
    
    def _get_memory_usage(self):
        """Return current process memory usage in MB."""
        try:
            p = psutil.Process(os.getpid())
            return p.memory_info().rss / (1024**2)
        except:
            return 0.0
    
    def _get_budget(self, benchmark_obj):
        """
        Determine iteration budget, try max_evals or fall back to dimension-based defaults.
        """
        if hasattr(benchmark_obj, 'max_evals'):
            return benchmark_obj.max_evals
        
        name_in_cfg = getattr(benchmark_obj, 'name', None)
        if name_in_cfg:
            budgets_map = self.config.get('budgets_by_benchmark', {})
            if name_in_cfg in budgets_map:
                return budgets_map[name_in_cfg]
        
        # dimension-based default
        try:
            dim = getattr(benchmark_obj, 'dim', len(benchmark_obj.bounds))
        except:
            dim = 5
        if dim <=5:
            return 30
        elif dim <=20:
            return 50
        else:
            return 100

def main():
    parser = argparse.ArgumentParser(description="Experiment Running Framework.")
    parser.add_argument("--config", type=str, default=,
                        help="Path to YAML config.")
    parser.add_argument("--run", action="store_true", help="Run experiments.")
    parser.add_argument("--analyze", action="store_true", help="Analyze results.")
    args = parser.parse_args()
    
    try:
        exp = Experiment(args.config)
        
        if args.run:
            results = exp.run_all()
            print("Experiments finished.")
        
        if args.analyze:
            loaded = exp._load_raw_results()
            exp.analyze_results(loaded)
            print("Analysis done.")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__=="__main__":
    main()

