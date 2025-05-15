# REBMBO: Reinforcement Energy-Based Model for Bayesian Optimization

This repository contains the implementation of REBMBO (Reinforcement Energy-Based Model for Bayesian Optimization), a novel approach that combines energy-based models with reinforcement learning for efficient black-box optimization.

## Project Overview

REBMBO enhances traditional Bayesian Optimization by introducing:
- Energy-based modeling for global structure learning
- Reinforcement learning for multi-step decision making
- Variants with different GP implementations (Classic, Sparse, Deep)

The project includes implementations of multiple baseline methods, benchmark problems, and a comprehensive experimental framework for evaluation and comparison.

## Run a single experiment with default settings
python src/main.py

## Specify a custom configuration
python src/main.py --config configs/experiments.yaml

## Run a specific method on a specific benchmark
python src/main.py --method rebmbo-c --benchmark branin

## Run with custom parameters
python src/main.py --method rebmbo-c --benchmark branin --iterations 50 --init_points 5

# Benchmarks
The repository includes implementations of various benchmark functions:
## Synthetic Benchmarks
- Branin (2D): Standard low-dimensional test function
- Rosenbrock (5D): Classic non-convex optimization test

## Real-world Benchmarks
- Nanophotonic Structure (3D): Optimization of photonic devices
- Rosetta Protein Design (86D): High-dimensional protein design task
- NATS-Bench (4D): Neural architecture search benchmark
- Robot Trajectory (6D): Robot motion planning optimization

# Methods
- REBMBO Variants

- REBMBO-C: Classic GP-based implementation
- REBMBO-S: Sparse GP variant for improved scalability
- REBMBO-D: Deep GP implementation for complex functions

## Baselines

- Classic BO: Standard Bayesian optimization
- TuRBO: Trust Region Bayesian Optimization
- EARL-BO: Ensemble Active Regression for BO
- BALLET-ICI: Method combining local and global search
- Two-Step EI: Two-step Expected Improvement
- KG: Knowledge Gradient
