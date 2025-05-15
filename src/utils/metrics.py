"""
Metrics Module for Optimization Evaluation.

This module provides functions for computing and analyzing various performance metrics
for optimization algorithms, including regret, convergence rate, and efficiency metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import matplotlib.pyplot as plt
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Metrics")


def compute_regret(y_values: List[float], 
                   optimal_value: float, 
                   maximize: bool = True) -> np.ndarray:
    """
    Compute regret for each iteration.
    
    Args:
        y_values (List[float]): Function values at each iteration
        optimal_value (float): The global optimum value
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
    
    Returns:
        np.ndarray: Regret at each iteration
    """
    y = np.array(y_values)
    
    if maximize:
        # For maximization: regret = optimal - current_best
        regret = optimal_value - y
    else:
        # For minimization: regret = current_best - optimal
        regret = y - optimal_value
    
    return np.maximum(regret, 0)  # Ensure non-negative regret


def compute_cumulative_regret(regret: np.ndarray) -> np.ndarray:
    """
    Compute cumulative regret over iterations.
    
    Args:
        regret (np.ndarray): Regret at each iteration
        
    Returns:
        np.ndarray: Cumulative regret at each iteration
    """
    return np.cumsum(regret)


def compute_simple_regret(y_values: List[float], 
                         optimal_value: float, 
                         maximize: bool = True) -> np.ndarray:
    """
    Compute simple regret (best value so far) for each iteration.
    
    Args:
        y_values (List[float]): Function values at each iteration
        optimal_value (float): The global optimum value
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
    
    Returns:
        np.ndarray: Simple regret at each iteration
    """
    y = np.array(y_values)
    
    if maximize:
        # Track best value seen so far for maximization
        best_y = np.maximum.accumulate(y)
        simple_regret = optimal_value - best_y
    else:
        # Track best value seen so far for minimization
        best_y = np.minimum.accumulate(y)
        simple_regret = best_y - optimal_value
    
    return np.maximum(simple_regret, 0)  # Ensure non-negative regret


def compute_pseudo_regret(y_values: List[float], 
                         best_y_values: List[float],
                         optimal_value: Optional[float] = None,
                         maximize: bool = True) -> np.ndarray:
    """
    Compute pseudo-regret, which uses running best value found so far.
    
    Args:
        y_values (List[float]): Function values at each iteration
        best_y_values (List[float]): Best function values found up to each iteration
        optimal_value (float, optional): The global optimum value (if known)
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
    
    Returns:
        np.ndarray: Pseudo-regret at each iteration
    """
    y = np.array(y_values)
    best_y = np.array(best_y_values)
    
    if optimal_value is not None:
        # If optimal value is known, compute regret relative to it
        if maximize:
            pseudo_regret = optimal_value - best_y
        else:
            pseudo_regret = best_y - optimal_value
    else:
        # If optimal value is unknown, use the best observed value as reference
        best_observed = best_y[-1]
        if maximize:
            pseudo_regret = best_observed - best_y
        else:
            pseudo_regret = best_y - best_observed
    
    return np.maximum(pseudo_regret, 0)  # Ensure non-negative regret


def compute_normalized_regret(regret: np.ndarray, 
                            initial_regret: float) -> np.ndarray:
    """
    Compute normalized regret (regret / initial regret).
    
    Args:
        regret (np.ndarray): Regret values
        initial_regret (float): Initial regret value
        
    Returns:
        np.ndarray: Normalized regret values
    """
    if initial_regret == 0:
        logger.warning("Initial regret is zero, cannot normalize")
        return np.zeros_like(regret)
    
    return regret / initial_regret


def compute_log_regret(regret: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute log of regret (log(regret + epsilon)).
    
    Args:
        regret (np.ndarray): Regret values
        epsilon (float): Small constant to avoid log(0)
        
    Returns:
        np.ndarray: Log regret values
    """
    return np.log(regret + epsilon)


def compute_convergence_rate(regret: np.ndarray, 
                            iterations: np.ndarray = None) -> Tuple[float, float]:
    """
    Estimate the convergence rate by fitting a power law to the regret.
    
    Args:
        regret (np.ndarray): Regret at each iteration
        iterations (np.ndarray, optional): Iteration indices
        
    Returns:
        Tuple[float, float]: Exponent and scale factor of the power law
    """
    if iterations is None:
        iterations = np.arange(1, len(regret) + 1)
    
    # Fit power law: regret ~ a * iteration^b
    # Using log-log regression
    log_iter = np.log(iterations)
    log_regret = np.log(regret + 1e-10)  # Add small constant to avoid log(0)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_iter, log_regret)
    
    return slope, np.exp(intercept)


def compute_iterations_to_threshold(regret: np.ndarray, 
                                  threshold: float) -> Optional[int]:
    """
    Compute the number of iterations needed to reach a regret threshold.
    
    Args:
        regret (np.ndarray): Regret at each iteration
        threshold (float): Regret threshold
        
    Returns:
        int or None: Number of iterations to reach threshold, or None if never reached
    """
    below_threshold = np.where(regret <= threshold)[0]
    if len(below_threshold) > 0:
        return below_threshold[0] + 1  # +1 because iterations are 1-indexed
    else:
        return None


def compute_auc(regret: np.ndarray, 
              max_iter: Optional[int] = None) -> float:
    """
    Compute area under the regret curve (AUC).
    
    Args:
        regret (np.ndarray): Regret at each iteration
        max_iter (int, optional): Maximum number of iterations to consider
        
    Returns:
        float: AUC score
    """
    if max_iter is not None and max_iter < len(regret):
        regret = regret[:max_iter]
    
    # AUC is just the sum of regret
    return float(np.sum(regret))


def compute_final_regret_statistics(regrets: List[np.ndarray], 
                                   at_iteration: Optional[int] = None) -> Dict[str, float]:
    """
    Compute statistics of final regret across multiple runs.
    
    Args:
        regrets (List[np.ndarray]): List of regret sequences from multiple runs
        at_iteration (int, optional): Specific iteration to evaluate (default: last common iteration)
        
    Returns:
        Dict[str, float]: Statistics of final regret
    """
    # Determine the iteration to evaluate
    if at_iteration is None:
        # Use the last common iteration across all runs
        min_length = min(len(r) for r in regrets)
        at_iteration = min_length - 1
    else:
        at_iteration = min(at_iteration, min(len(r) for r in regrets)) - 1
    
    # Extract final regret values
    final_regrets = [r[at_iteration] for r in regrets]
    
    # Compute statistics
    stats = {
        'mean': float(np.mean(final_regrets)),
        'std': float(np.std(final_regrets)),
        'min': float(np.min(final_regrets)),
        'max': float(np.max(final_regrets)),
        'median': float(np.median(final_regrets)),
        '25th_percentile': float(np.percentile(final_regrets, 25)),
        '75th_percentile': float(np.percentile(final_regrets, 75)),
    }
    
    return stats


def compute_efficiency_metrics(regrets: List[np.ndarray], 
                             times: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute efficiency metrics (regret per unit time).
    
    Args:
        regrets (List[np.ndarray]): List of regret sequences from multiple runs
        times (List[np.ndarray]): List of cumulative time sequences from multiple runs
        
    Returns:
        Dict[str, float]: Efficiency metrics
    """
    # Ensure regrets and times have the same length
    assert len(regrets) == len(times), "regrets and times must have the same length"
    
    # Compute regret reduction per unit time for each run
    efficiency_metrics = []
    
    for regret, time in zip(regrets, times):
        # Use the minimum common length
        min_length = min(len(regret), len(time))
        regret = regret[:min_length]
        time = time[:min_length]
        
        # Skip runs with no time elapsed
        if np.sum(time) == 0:
            continue
        
        # Initial regret
        initial_regret = regret[0]
        
        # Regret reduction (initial - final)
        regret_reduction = initial_regret - regret[-1]
        
        # Total time
        total_time = np.sum(time)
        
        # Efficiency = regret reduction per unit time
        efficiency = regret_reduction / total_time
        
        efficiency_metrics.append(efficiency)
    
    # Compute statistics
    if not efficiency_metrics:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    stats = {
        'mean': float(np.mean(efficiency_metrics)),
        'std': float(np.std(efficiency_metrics)),
        'min': float(np.min(efficiency_metrics)),
        'max': float(np.max(efficiency_metrics)),
        'median': float(np.median(efficiency_metrics)),
    }
    
    return stats


def compute_relative_performance(regrets_a: List[np.ndarray], 
                               regrets_b: List[np.ndarray],
                               at_iteration: Optional[int] = None) -> Dict[str, float]:
    """
    Compare the relative performance of two methods (A vs B).
    
    Args:
        regrets_a (List[np.ndarray]): Regret sequences for method A
        regrets_b (List[np.ndarray]): Regret sequences for method B
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        Dict[str, float]: Relative performance metrics
    """
    # Get statistics for both methods
    stats_a = compute_final_regret_statistics(regrets_a, at_iteration)
    stats_b = compute_final_regret_statistics(regrets_b, at_iteration)
    
    # Compute relative metrics
    if stats_b['mean'] == 0:
        rel_mean = float('inf') if stats_a['mean'] > 0 else 1.0
    else:
        rel_mean = stats_a['mean'] / stats_b['mean']
    
    # For statistical testing, extract final regrets
    if at_iteration is None:
        min_length_a = min(len(r) for r in regrets_a)
        min_length_b = min(len(r) for r in regrets_b)
        it_a = min_length_a - 1
        it_b = min_length_b - 1
    else:
        it_a = min(at_iteration, min(len(r) for r in regrets_a)) - 1
        it_b = min(at_iteration, min(len(r) for r in regrets_b)) - 1
    
    final_regrets_a = [r[it_a] for r in regrets_a]
    final_regrets_b = [r[it_b] for r in regrets_b]
    
    # Perform Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(final_regrets_a, final_regrets_b, alternative='less')
    except ValueError:
        # If test fails (e.g., due to identical values), set default values
        u_stat, p_value = 0, 1.0
    
    # Relative performance metrics
    metrics = {
        'rel_mean': rel_mean,
        'method_a_mean': stats_a['mean'],
        'method_b_mean': stats_b['mean'],
        'improvement': (stats_b['mean'] - stats_a['mean']) / stats_b['mean'] if stats_b['mean'] != 0 else 0.0,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
    
    return metrics


def compute_average_rank(methods_regrets: Dict[str, List[np.ndarray]],
                        at_iteration: Optional[int] = None) -> Dict[str, float]:
    """
    Compute the average rank of each method across multiple runs.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        Dict[str, float]: Average rank of each method (lower is better)
    """
    # Extract method names and regret sequences
    method_names = list(methods_regrets.keys())
    regrets_lists = list(methods_regrets.values())
    
    # Determine the number of runs (assume all methods have the same number)
    n_runs = len(regrets_lists[0])
    
    # Initialize ranks
    ranks = {method: [] for method in method_names}
    
    # For each run, rank the methods
    for run_idx in range(n_runs):
        # Extract final regret for each method
        if at_iteration is None:
            # Use the last common iteration
            min_lengths = [min(len(r) for r in method_regrets) for method_regrets in regrets_lists]
            final_regrets = [method_regrets[run_idx][min_lengths[i] - 1] 
                            for i, method_regrets in enumerate(regrets_lists)]
        else:
            # Use the specified iteration
            final_regrets = [method_regrets[run_idx][min(at_iteration, len(method_regrets[run_idx])) - 1] 
                            for method_regrets in regrets_lists]
        
        # Rank the methods (lower regret -> lower rank -> better)
        sorted_indices = np.argsort(final_regrets)
        method_ranks = np.empty_like(sorted_indices)
        method_ranks[sorted_indices] = np.arange(len(sorted_indices))
        
        # Store ranks
        for i, method in enumerate(method_names):
            ranks[method].append(method_ranks[i])
    
    # Compute average rank for each method
    average_ranks = {method: float(np.mean(ranks[method])) for method in method_names}
    
    return average_ranks


def compute_friedman_test(methods_regrets: Dict[str, List[np.ndarray]],
                         at_iteration: Optional[int] = None) -> Tuple[float, float]:
    """
    Perform Friedman test to determine if there are significant differences among methods.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        Tuple[float, float]: Friedman statistic and p-value
    """
    # Extract method names and regret sequences
    method_names = list(methods_regrets.keys())
    regrets_lists = list(methods_regrets.values())
    
    # Determine the number of runs and methods
    n_runs = len(regrets_lists[0])
    n_methods = len(method_names)
    
    # Initialize matrix of final regrets: rows=runs, cols=methods
    final_regrets = np.zeros((n_runs, n_methods))
    
    # Fill the matrix
    for run_idx in range(n_runs):
        for method_idx, method_regrets in enumerate(regrets_lists):
            if at_iteration is None:
                # Use the last iteration
                final_regrets[run_idx, method_idx] = method_regrets[run_idx][-1]
            else:
                # Use the specified iteration
                it = min(at_iteration, len(method_regrets[run_idx])) - 1
                final_regrets[run_idx, method_idx] = method_regrets[run_idx][it]
    
    # Perform Friedman test
    try:
        friedman_stat, p_value = stats.friedmanchisquare(*[final_regrets[:, i] for i in range(n_methods)])
    except ValueError as e:
        logger.error(f"Error in Friedman test: {e}")
        return 0.0, 1.0
    
    return float(friedman_stat), float(p_value)


def compute_nemenyi_test(methods_regrets: Dict[str, List[np.ndarray]],
                        at_iteration: Optional[int] = None) -> Dict[Tuple[str, str], float]:
    """
    Perform Nemenyi post-hoc test for pairwise comparisons after Friedman test.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        Dict[Tuple[str, str], float]: p-values for each method pair
    """
    # This is a simplified version. For a proper implementation, a specialized library
    # like scikit-posthocs should be used.
    
    # Compute average ranks
    avg_ranks = compute_average_rank(methods_regrets, at_iteration)
    
    # Number of runs and methods
    n_runs = len(next(iter(methods_regrets.values())))
    n_methods = len(methods_regrets)
    
    # Critical difference
    q_alpha = 2.0  # Approximation for 95% confidence level
    critical_diff = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_runs))
    
    # Pairwise p-values (simplified)
    p_values = {}
    method_names = list(methods_regrets.keys())
    
    for i in range(n_methods):
        for j in range(i+1, n_methods):
            method_i = method_names[i]
            method_j = method_names[j]
            
            # Compute absolute rank difference
            rank_diff = abs(avg_ranks[method_i] - avg_ranks[method_j])
            
            # Simplified p-value based on critical difference
            p_value = 1.0 if rank_diff < critical_diff else 0.01
            
            p_values[(method_i, method_j)] = p_value
    
    return p_values


def compute_score_matrix(methods_regrets: Dict[str, List[np.ndarray]],
                        at_iterations: List[int] = None) -> pd.DataFrame:
    """
    Compute a score matrix for multiple methods and iterations.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iterations (List[int], optional): List of iterations to evaluate
        
    Returns:
        pd.DataFrame: Score matrix (method x iteration)
    """
    method_names = list(methods_regrets.keys())
    
    # Determine iterations to evaluate if not provided
    if at_iterations is None:
        # Find the minimum length across all methods and runs
        min_length = float('inf')
        for method_regrets in methods_regrets.values():
            for run_regrets in method_regrets:
                min_length = min(min_length, len(run_regrets))
        
        # Use a few evenly spaced iterations
        n_samples = 5
        at_iterations = np.linspace(1, min_length, n_samples, dtype=int).tolist()
    
    # Initialize score matrix
    scores = np.zeros((len(method_names), len(at_iterations)))
    
    # Fill the matrix with mean final regret for each method and iteration
    for method_idx, method in enumerate(method_names):
        regrets = methods_regrets[method]
        for iter_idx, iteration in enumerate(at_iterations):
            # Compute mean regret at this iteration
            it = min(iteration, min(len(r) for r in regrets)) - 1
            final_regrets = [r[it] for r in regrets]
            scores[method_idx, iter_idx] = np.mean(final_regrets)
    
    # Create DataFrame
    score_df = pd.DataFrame(scores, index=method_names, columns=[f"Iter {i}" for i in at_iterations])
    
    return score_df


def compute_pairwise_win_rate(methods_regrets: Dict[str, List[np.ndarray]],
                            at_iteration: Optional[int] = None) -> pd.DataFrame:
    """
    Compute pairwise win rates between methods.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        pd.DataFrame: Win rate matrix (method_a x method_b)
    """
    method_names = list(methods_regrets.keys())
    n_methods = len(method_names)
    
    # Initialize win rate matrix
    win_rates = np.zeros((n_methods, n_methods))
    
    # Determine the number of runs
    n_runs = len(next(iter(methods_regrets.values())))
    
    # For each pair of methods, count wins
    for i in range(n_methods):
        for j in range(n_methods):
            if i == j:
                # Diagonal elements are 0.5 (tie against itself)
                win_rates[i, j] = 0.5
                continue
            
            method_i = method_names[i]
            method_j = method_names[j]
            
            wins = 0
            
            # Compare across runs
            for run in range(n_runs):
                regrets_i = methods_regrets[method_i][run]
                regrets_j = methods_regrets[method_j][run]
                
                if at_iteration is None:
                    # Use the last common iteration
                    min_length = min(len(regrets_i), len(regrets_j))
                    it = min_length - 1
                else:
                    # Use the specified iteration
                    it = min(at_iteration, min(len(regrets_i), len(regrets_j))) - 1
                
                # Method i wins if its regret is lower
                if regrets_i[it] < regrets_j[it]:
                    wins += 1
                elif regrets_i[it] == regrets_j[it]:
                    # Tie counts as half a win
                    wins += 0.5
            
            # Win rate
            win_rates[i, j] = wins / n_runs
    
    # Create DataFrame
    win_rate_df = pd.DataFrame(win_rates, index=method_names, columns=method_names)
    
    return win_rate_df


def summarize_optimization_results(results: Dict[str, Dict], 
                                 optimal_values: Dict[str, float] = None,
                                 maximize: bool = True) -> pd.DataFrame:
    """
    Create a summary DataFrame of optimization results for multiple methods and benchmarks.
    
    Args:
        results (Dict[str, Dict]): Results dictionary: {benchmark: {method: {'history': {...}}}}
        optimal_values (Dict[str, float], optional): Known optimal values for each benchmark
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
        
    Returns:
        pd.DataFrame: Summary DataFrame
    """
    # Initialize data for the DataFrame
    data = []
    
    # For each benchmark and method
    for benchmark, methods in results.items():
        # Get optimal value if available
        optimal_value = None
        if optimal_values is not None and benchmark in optimal_values:
            optimal_value = optimal_values[benchmark]
        
        for method, res in methods.items():
            # Extract history
            history = res.get('history', {})
            
            # Skip if history is empty
            if not history or 'y' not in history or not history['y']:
                continue
            
            # Extract metrics
            y_values = history['y']
            best_y_values = history.get('best_y', None)
            
            # If best_y not available, compute it
            if best_y_values is None:
                if maximize:
                    best_y_values = np.maximum.accumulate(y_values)
                else:
                    best_y_values = np.minimum.accumulate(y_values)
            
            # Compute final regret if optimal value is known
            final_regret = None
            if optimal_value is not None:
                if maximize:
                    final_regret = optimal_value - best_y_values[-1]
                else:
                    final_regret = best_y_values[-1] - optimal_value
                
                # Ensure non-negative regret
                final_regret = max(0, final_regret)
            
            # Extract other metrics
            n_iterations = len(y_values)
            best_value = best_y_values[-1]
            
            # Extract time metrics if available
            time_per_iter = None
            if 'time' in history and history['time']:
                time_per_iter = np.mean(history['time'])
            
            # Extract model time if available
            model_time = None
            if 'model_time' in history and history['model_time']:
                model_time = np.mean(history['model_time'])
            
            # Add row to data
            data.append({
                'Benchmark': benchmark,
                'Method': method,
                'Best Value': best_value,
                'Final Regret': final_regret,
                'Iterations': n_iterations,
                'Time per Iteration (s)': time_per_iter,
                'Model Update Time (s)': model_time
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def compute_method_ranking(results: Dict[str, Dict], 
                         optimal_values: Dict[str, float] = None,
                         maximize: bool = True) -> pd.DataFrame:
    """
    Compute a ranking of methods across benchmarks.
    
    Args:
        results (Dict[str, Dict]): Results dictionary: {benchmark: {method: {'history': {...}}}}
        optimal_values (Dict[str, float], optional): Known optimal values for each benchmark
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
        
    Returns:
        pd.DataFrame: Method ranking DataFrame
    """
    # Initialize data for the DataFrame
    benchmark_names = list(results.keys())
    method_names = set()
    
    # Collect all method names
    for methods in results.values():
        method_names.update(methods.keys())
    
    method_names = list(method_names)
    
    # Initialize rank matrix
    rank_matrix = np.zeros((len(method_names), len(benchmark_names)))
    
    # Compute ranks for each benchmark
    for bench_idx, benchmark in enumerate(benchmark_names):
        methods = results[benchmark]
        
        # Extract final best values for each method
        best_values = {}
        for method, res in methods.items():
            history = res.get('history', {})
            if not history or 'best_y' not in history or not history['best_y']:
                continue
            
            best_values[method] = history['best_y'][-1]
        
        # Skip if no best values available
        if not best_values:
            continue
        
        # Rank methods (higher best value -> lower rank for maximization)
        # (lower best value -> lower rank for minimization)
        method_list = list(best_values.keys())
        values_list = [best_values[m] for m in method_list]
        
        # Sort indices (reversed for maximization)
        if maximize:
            # Higher is better for maximization
            sorted_indices = np.argsort(-np.array(values_list))
        else:
            # Lower is better for minimization
            sorted_indices = np.argsort(np.array(values_list))
        
        # Assign ranks (1-indexed)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        
        # Fill rank matrix
        for i, method in enumerate(method_list):
            method_idx = method_names.index(method)
            rank_matrix[method_idx, bench_idx] = ranks[i]
    
    # Compute average rank for each method
    avg_ranks = np.zeros(len(method_names))
    rank_counts = np.zeros(len(method_names))
    
    for i in range(len(method_names)):
        # Consider only benchmarks where the method was evaluated
        valid_ranks = rank_matrix[i, :][rank_matrix[i, :] > 0]
        if len(valid_ranks) > 0:
            avg_ranks[i] = np.mean(valid_ranks)
            rank_counts[i] = len(valid_ranks)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Method': method_names,
        'Average Rank': avg_ranks,
        'Benchmarks Completed': rank_counts,
    })
    
    # Sort by average rank
    df = df.sort_values('Average Rank')
    
    return df

def compute_method_ranking(results: Dict[str, Dict], 
                         optimal_values: Dict[str, float] = None,
                         maximize: bool = True) -> pd.DataFrame:
    """
    Compute a ranking of methods across benchmarks.
    
    Args:
        results (Dict[str, Dict]): Results dictionary: {benchmark: {method: {'history': {...}}}}
        optimal_values (Dict[str, float], optional): Known optimal values for each benchmark
        maximize (bool): Whether the objective is maximization (True) or minimization (False)
        
    Returns:
        pd.DataFrame: Method ranking DataFrame
    """
    # Initialize data for the DataFrame
    benchmark_names = list(results.keys())
    method_names = set()
    
    # Collect all method names
    for methods in results.values():
        method_names.update(methods.keys())
    
    method_names = list(method_names)
    
    # Initialize rank matrix
    rank_matrix = np.zeros((len(method_names), len(benchmark_names)))
    
    # Compute ranks for each benchmark
    for bench_idx, benchmark in enumerate(benchmark_names):
        methods = results[benchmark]
        
        # Extract final best values for each method
        best_values = {}
        for method, res in methods.items():
            history = res.get('history', {})
            if not history or 'best_y' not in history or not history['best_y']:
                continue
            
            best_values[method] = history['best_y'][-1]
        
        # Skip if no best values available
        if not best_values:
            continue
        
        # Rank methods (higher best value -> lower rank for maximization)
        # (lower best value -> lower rank for minimization)
        method_list = list(best_values.keys())
        values_list = [best_values[m] for m in method_list]
        
        # Sort indices (reversed for maximization)
        if maximize:
            # Higher is better for maximization
            sorted_indices = np.argsort(-np.array(values_list))
        else:
            # Lower is better for minimization
            sorted_indices = np.argsort(np.array(values_list))
        
        # Assign ranks (1-indexed)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        
        # Fill rank matrix
        for i, method in enumerate(method_list):
            method_idx = method_names.index(method)
            rank_matrix[method_idx, bench_idx] = ranks[i]
    
    # Compute average rank for each method
    avg_ranks = np.zeros(len(method_names))
    rank_counts = np.zeros(len(method_names))
    
    for i in range(len(method_names)):
        # Consider only benchmarks where the method was evaluated
        valid_ranks = rank_matrix[i, :][rank_matrix[i, :] > 0]
        if len(valid_ranks) > 0:
            avg_ranks[i] = np.mean(valid_ranks)
            rank_counts[i] = len(valid_ranks)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Method': method_names,
        'Average Rank': avg_ranks,
        'Benchmarks Completed': rank_counts,
    })
    
    # Sort by average rank
    df = df.sort_values('Average Rank')
    
    return df


def format_optimization_summary(df: pd.DataFrame) -> str:
    """
    Format optimization summary as a formatted string.
    
    Args:
        df (pd.DataFrame): Summary DataFrame from summarize_optimization_results
        
    Returns:
        str: Formatted summary string
    """
    if df.empty:
        return "No optimization results available."
    
    # Build formatted string
    lines = []
    lines.append("Optimization Results Summary")
    lines.append("===========================")
    lines.append("")
    
    # Group by benchmark
    for benchmark, group in df.groupby('Benchmark'):
        lines.append(f"Benchmark: {benchmark}")
        lines.append("-" * (len(benchmark) + 11))
        
        # Sort by best value (descending)
        group = group.sort_values('Best Value', ascending=False)
        
        # Format results
        for _, row in group.iterrows():
            method = row['Method']
            best_value = row['Best Value']
            final_regret = row.get('Final Regret', 'N/A')
            iterations = row['Iterations']
            time_per_iter = row.get('Time per Iteration (s)', 'N/A')
            
            # Format values
            if isinstance(best_value, (int, float)):
                best_value = f"{best_value:.6f}"
            if isinstance(final_regret, (int, float)):
                final_regret = f"{final_regret:.6f}"
            if isinstance(time_per_iter, (int, float)):
                time_per_iter = f"{time_per_iter:.4f}s"
            
            lines.append(f"  {method}:")
            lines.append(f"    Best Value: {best_value}")
            lines.append(f"    Final Regret: {final_regret}")
            lines.append(f"    Iterations: {iterations}")
            lines.append(f"    Time per Iteration: {time_per_iter}")
            lines.append("")
        
        lines.append("")
    
    return "\n".join(lines)


def format_ranking_table(df: pd.DataFrame) -> str:
    """
    Format method ranking as a formatted string.
    
    Args:
        df (pd.DataFrame): Ranking DataFrame from compute_method_ranking
        
    Returns:
        str: Formatted ranking table string
    """
    if df.empty:
        return "No ranking available."
    
    # Build formatted string
    lines = []
    lines.append("Method Ranking")
    lines.append("=============")
    lines.append("")
    lines.append("Rank | Method | Average Rank | Benchmarks")
    lines.append("-" * 50)
    
    # Format each row
    for i, (_, row) in enumerate(df.iterrows()):
        method = row['Method']
        avg_rank = row['Average Rank']
        benchmarks = int(row['Benchmarks Completed'])
        
        lines.append(f"{i+1:4d} | {method:20s} | {avg_rank:11.2f} | {benchmarks:10d}")
    
    lines.append("")
    
    return "\n".join(lines)


def plot_regret_curves(methods_regrets: Dict[str, List[np.ndarray]],
                     title: str = "Regret Curves",
                     log_scale: bool = False,
                     max_iter: Optional[int] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot regret curves for multiple methods.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        title (str): Plot title
        log_scale (bool): Whether to use log scale for y-axis
        max_iter (int, optional): Maximum iteration to plot
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot regret curves for each method
    for method, regrets in methods_regrets.items():
        # Compute mean and std of regret across runs
        min_length = min(len(r) for r in regrets)
        if max_iter is not None:
            min_length = min(min_length, max_iter)
        
        # Truncate regrets to min_length
        truncated_regrets = [r[:min_length] for r in regrets]
        
        # Compute mean and std
        mean_regret = np.mean(truncated_regrets, axis=0)
        std_regret = np.std(truncated_regrets, axis=0)
        
        # Plot mean and confidence interval
        x = np.arange(1, min_length + 1)
        ax.plot(x, mean_regret, label=method)
        ax.fill_between(x, mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)
    
    # Set plot properties
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Use log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_convergence_comparison(methods_regrets: Dict[str, List[np.ndarray]],
                              thresholds: List[float],
                              title: str = "Convergence Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comparison of iterations to reach different regret thresholds.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        thresholds (List[float]): List of regret thresholds
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Compute iterations to reach each threshold for each method
    methods = list(methods_regrets.keys())
    n_methods = len(methods)
    n_thresholds = len(thresholds)
    
    # Initialize data
    iterations_to_threshold = np.zeros((n_methods, n_thresholds))
    iterations_std = np.zeros((n_methods, n_thresholds))
    
    for method_idx, method in enumerate(methods):
        regrets = methods_regrets[method]
        
        for threshold_idx, threshold in enumerate(thresholds):
            # Compute iterations to reach threshold for each run
            iters = []
            
            for regret in regrets:
                iter_to_threshold = compute_iterations_to_threshold(regret, threshold)
                if iter_to_threshold is not None:
                    iters.append(iter_to_threshold)
            
            # Compute mean and std
            if iters:
                iterations_to_threshold[method_idx, threshold_idx] = np.mean(iters)
                iterations_std[method_idx, threshold_idx] = np.std(iters)
            else:
                # If threshold never reached, set to NaN
                iterations_to_threshold[method_idx, threshold_idx] = np.nan
                iterations_std[method_idx, threshold_idx] = np.nan
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Width of bars
    width = 0.8 / n_methods
    
    # Position of bars
    positions = np.arange(n_thresholds)
    
    # Plot bars for each method
    for method_idx, method in enumerate(methods):
        pos = positions + (method_idx - n_methods/2 + 0.5) * width
        
        bars = ax.bar(pos, iterations_to_threshold[method_idx], width,
                     label=method, alpha=0.7)
        
        # Add error bars
        ax.errorbar(pos, iterations_to_threshold[method_idx],
                  yerr=iterations_std[method_idx],
                  fmt='none', capsize=5, color='black', alpha=0.5)
    
    # Set plot properties
    ax.set_xlabel("Regret Threshold")
    ax.set_ylabel("Iterations to Threshold")
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{t:.3f}" for t in thresholds])
    ax.legend()
    ax.grid(True, axis='y')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_win_rate_heatmap(win_rate_df: pd.DataFrame,
                        title: str = "Win Rate Matrix",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot win rate matrix as a heatmap.
    
    Args:
        win_rate_df (pd.DataFrame): Win rate DataFrame from compute_pairwise_win_rate
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(win_rate_df.values, cmap='Blues', vmin=0, vmax=1)
    
    # Set plot properties
    ax.set_title(title)
    ax.set_xlabel("Method B")
    ax.set_ylabel("Method A")
    
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(win_rate_df.columns)))
    ax.set_yticks(np.arange(len(win_rate_df.index)))
    ax.set_xticklabels(win_rate_df.columns)
    ax.set_yticklabels(win_rate_df.index)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Win Rate", rotation=-90, va="bottom")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(win_rate_df.index)):
        for j in range(len(win_rate_df.columns)):
            text = ax.text(j, i, f"{win_rate_df.iloc[i, j]:.2f}",
                         ha="center", va="center", color="black" if win_rate_df.iloc[i, j] < 0.5 else "white")
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_rank_distribution(methods_regrets: Dict[str, List[np.ndarray]],
                         at_iterations: List[int],
                         title: str = "Rank Distribution",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of ranks for each method across iterations.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iterations (List[int]): List of iterations to evaluate
        title (str): Plot title
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Compute average rank for each method at each iteration
    avg_ranks = {}
    
    for it in at_iterations:
        avg_ranks[it] = compute_average_rank(methods_regrets, it)
    
    # Convert to DataFrame
    methods = list(methods_regrets.keys())
    data = []
    
    for method in methods:
        for it in at_iterations:
            data.append({
                'Method': method,
                'Iteration': it,
                'Average Rank': avg_ranks[it][method]
            })
    
    df = pd.DataFrame(data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot rank distribution for each method
    for method in methods:
        method_data = df[df['Method'] == method]
        ax.plot(method_data['Iteration'], method_data['Average Rank'], 'o-', label=method)
    
    # Set plot properties
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Rank")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Invert y-axis (lower rank is better)
    ax.invert_yaxis()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_performance_profile(methods_regrets: Dict[str, List[np.ndarray]],
                              at_iteration: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Compute performance profile data for multiple methods.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        
    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]: Performance profile data and tau values
    """
    # Extract method names and regret sequences
    method_names = list(methods_regrets.keys())
    
    # Determine the best performance for each run
    best_performances = []
    performances = {}
    
    # Initialize performances dictionary
    for method in method_names:
        performances[method] = []
    
    # Extract final regret for each run and method
    n_runs = len(next(iter(methods_regrets.values())))
    
    for run_idx in range(n_runs):
        run_performances = []
        
        for method in method_names:
            regrets = methods_regrets[method]
            
            if at_iteration is None:
                # Use the last iteration
                final_regret = regrets[run_idx][-1]
            else:
                # Use the specified iteration
                it = min(at_iteration, len(regrets[run_idx])) - 1
                final_regret = regrets[run_idx][it]
            
            performances[method].append(final_regret)
            run_performances.append(final_regret)
        
        # Find best performance for this run
        best_performances.append(min(run_performances))
    
    # Compute performance ratios
    ratios = {}
    
    for method in method_names:
        method_performances = performances[method]
        ratios[method] = np.array([perf / best for perf, best in zip(method_performances, best_performances)])
    
    # Generate tau values
    tau_values = np.linspace(1, 10, 100)
    
    # Compute performance profile data
    pp_data = {}
    
    for method in method_names:
        method_ratios = ratios[method]
        pp_data[method] = np.array([np.mean(method_ratios <= tau) for tau in tau_values])
    
    return pp_data, tau_values


def plot_performance_profile(methods_regrets: Dict[str, List[np.ndarray]],
                           at_iteration: Optional[int] = None,
                           title: str = "Performance Profile",
                           log_scale: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot performance profile for multiple methods.
    
    Args:
        methods_regrets (Dict[str, List[np.ndarray]]): Dictionary mapping method names to lists of regret sequences
        at_iteration (int, optional): Specific iteration to evaluate
        title (str): Plot title
        log_scale (bool): Whether to use log scale for x-axis
        save_path (str, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    # Compute performance profile data
    pp_data, tau_values = compute_performance_profile(methods_regrets, at_iteration)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot performance profile for each method
    for method, data in pp_data.items():
        ax.plot(tau_values, data, label=method)
    
    # Set plot properties
    ax.set_xlabel("Performance Ratio (τ)")
    ax.set_ylabel("Probability P(r_p,s ≤ τ)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    # Set limits
    ax.set_xlim([1, tau_values[-1]])
    ax.set_ylim([0, 1.05])
    
    # Use log scale if requested
    if log_scale:
        ax.set_xscale('log')
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def test_metrics_module():
    """
    Test the metrics module with synthetic data.
    """
    # Create synthetic data
    np.random.seed(42)
    
    n_methods = 3
    n_runs = 5
    n_iterations = 50
    
    # Generate synthetic regret sequences
    methods_regrets = {}
    
    methods = ["Method A", "Method B", "Method C"]
    
    for i, method in enumerate(methods):
        # Create regret sequences with different convergence rates
        regrets = []
        
        for run in range(n_runs):
            # Generate synthetic regret sequence
            # Method A converges quickly, Method B moderately, Method C slowly
            rate = 0.1 * (i + 1)
            noise = 0.1 * (run + 1)
            
            regret = np.array([1.0 * np.exp(-rate * t) + noise * np.random.rand() for t in range(n_iterations)])
            regrets.append(regret)
        
        methods_regrets[method] = regrets
    
    # Test basic metrics
    print("Testing basic metrics...")
    
    # Example regret sequence
    regret = methods_regrets["Method A"][0]
    
    # Compute various metrics
    simple_regret = compute_simple_regret(1.0 - regret, 1.0)
    cumulative_regret = compute_cumulative_regret(regret)
    normalized_regret = compute_normalized_regret(regret, regret[0])
    log_regret = compute_log_regret(regret)
    rate, scale = compute_convergence_rate(regret)
    
    print(f"Simple regret shape: {simple_regret.shape}")
    print(f"Cumulative regret shape: {cumulative_regret.shape}")
    print(f"Normalized regret shape: {normalized_regret.shape}")
    print(f"Log regret shape: {log_regret.shape}")
    print(f"Convergence rate: {rate:.4f}, scale: {scale:.4f}")
    
    # Test iterations to threshold
    threshold = 0.1
    iters = compute_iterations_to_threshold(regret, threshold)
    print(f"Iterations to threshold {threshold}: {iters}")
    
    # Test AUC
    auc = compute_auc(regret)
    print(f"AUC: {auc:.4f}")
    
    # Test final regret statistics
    stats = compute_final_regret_statistics(methods_regrets["Method A"])
    print("Final regret statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test efficiency metrics
    times = [np.ones_like(r) for r in methods_regrets["Method A"]]
    efficiency = compute_efficiency_metrics(methods_regrets["Method A"], times)
    print("Efficiency metrics:")
    for key, value in efficiency.items():
        print(f"  {key}: {value:.4f}")
    
    # Test relative performance
    rel_perf = compute_relative_performance(methods_regrets["Method A"], methods_regrets["Method B"])
    print("Relative performance:")
    for key, value in rel_perf.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test average rank
    avg_ranks = compute_average_rank(methods_regrets)
    print("Average ranks:")
    for method, rank in avg_ranks.items():
        print(f"  {method}: {rank:.4f}")
    
    # Test Friedman test
    friedman_stat, p_value = compute_friedman_test(methods_regrets)
    print(f"Friedman test: stat={friedman_stat:.4f}, p-value={p_value:.4f}")
    
    # Test Nemenyi test
    nemenyi_p_values = compute_nemenyi_test(methods_regrets)
    print("Nemenyi test p-values:")
    for pair, p_value in nemenyi_p_values.items():
        print(f"  {pair}: {p_value:.4f}")
    
    # Test score matrix
    score_matrix = compute_score_matrix(methods_regrets)
    print("Score matrix:")
    print(score_matrix)
    
    # Test pairwise win rate
    win_rate = compute_pairwise_win_rate(methods_regrets)
    print("Pairwise win rate:")
    print(win_rate)
    
    # Test performance profile
    pp_data, tau_values = compute_performance_profile(methods_regrets)
    print("Performance profile data:")
    for method, data in pp_data.items():
        print(f"  {method}: {data.shape}")
    
    print("All tests completed successfully!")
    
    return methods_regrets


if __name__ == "__main__":
    # Test the metrics module
    methods_regrets = test_metrics_module()
    
    # Create a few plots for visualization
    plt.figure(figsize=(12, 8))
    
    # Plot regret curves
    plt.subplot(2, 2, 1)
    for method, regrets in methods_regrets.items():
        mean_regret = np.mean(regrets, axis=0)
        plt.plot(mean_regret, label=method)
    plt.title("Mean Regret")
    plt.xlabel("Iteration")
    plt.ylabel("Regret")
    plt.legend()
    plt.grid(True)
    
    # Plot win rate heatmap
    plt.subplot(2, 2, 2)
    win_rate = compute_pairwise_win_rate(methods_regrets)
    plt.imshow(win_rate.values, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label="Win Rate")
    plt.title("Win Rate Heatmap")
    plt.xticks(np.arange(len(win_rate.columns)), win_rate.columns, rotation=45)
    plt.yticks(np.arange(len(win_rate.index)), win_rate.index)
    
    # Plot performance profile
    plt.subplot(2, 2, 3)
    pp_data, tau_values = compute_performance_profile(methods_regrets)
    for method, data in pp_data.items():
        plt.plot(tau_values, data, label=method)
    plt.title("Performance Profile")
    plt.xlabel("Performance Ratio (τ)")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    
    # Plot score matrix as a heatmap
    plt.subplot(2, 2, 4)
    score_matrix = compute_score_matrix(methods_regrets)
    plt.imshow(score_matrix.values, cmap='YlOrRd_r')
    plt.colorbar(label="Regret")
    plt.title("Score Matrix")
    plt.xticks(np.arange(len(score_matrix.columns)), score_matrix.columns, rotation=45)
    plt.yticks(np.arange(len(score_matrix.index)), score_matrix.index)
    
    plt.tight_layout()
    plt.savefig("metrics_test.png", dpi=300, bbox_inches='tight')
    print("Saved test plots to metrics_test.png")