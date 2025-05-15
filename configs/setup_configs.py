import os
import yaml

# Create the necessary directories
def create_directories():
    os.makedirs('./configs/methods', exist_ok=True)
    print("Created directory: ./configs/methods")

# Convert REBMBO configurations to YAML
def create_rebmbo_config():
    # These are the REBMBO configurations from your second file
    rebmbo_config = {
        "common": {
            "acquisition_type": "ebm_ucb",
            "initial_points": 5,
            "exploration_weight": 0.1,
            "verbose": True,
            "normalize_inputs": True,
            "seed": 42
        },
        "rebmbo_classic": {
            "name": "REBMBO-C",
            "type": "rebmbo",
            "variant": "classic",
            "gp_params": {
                "kernel": "matern_rbf_mix",
                "kernel_params": {
                    "matern_nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10,
                "optimizer": "L-BFGS-B",
                "n_restarts_optimizer": 5,
                "alpha": 1.0e-10
            },
            "ebm_params": {
                "hidden_layers": [64, 64],
                "learning_rate": 0.001,
                "mcmc_steps": 20,
                "mcmc_step_size": 0.1,
                "mcmc_noise_scale": 0.01,
                "lambda": 0.1,
                "update_frequency": 5,
                "batch_size": 32,
                "epochs_per_update": 10,
                "normalize_energy": True
            },
            "ppo_params": {
                "hidden_layers": [128, 64],
                "actor_learning_rate": 3.0e-4,
                "critic_learning_rate": 1.0e-3,
                "gamma": 0.99,
                "lambda_gae": 0.95,
                "clip_ratio": 0.2,
                "target_kl": 0.01,
                "max_grad_norm": 0.5,
                "update_epochs": 10,
                "batch_size": 32,
                "action_std_init": 0.6,
                "action_std_decay_rate": 0.05,
                "min_action_std": 0.1,
                "state_normalize": True
            }
        },
        "rebmbo_sparse": {
            "name": "REBMBO-S",
            "type": "rebmbo",
            "variant": "sparse",
            "gp_params": {
                "kernel": "matern_rbf_mix",
                "kernel_params": {
                    "matern_nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "inducing_points": 50,
                "inducing_point_method": "kmeans",
                "normalize_y": True,
                "optimization_restarts": 10,
                "alpha": 1.0e-10
            },
            "ebm_params": {
                "hidden_layers": [64, 64],
                "learning_rate": 0.001,
                "mcmc_steps": 20,
                "mcmc_step_size": 0.1,
                "mcmc_noise_scale": 0.01,
                "lambda": 0.1,
                "update_frequency": 5,
                "batch_size": 32,
                "epochs_per_update": 10,
                "normalize_energy": True
            },
            "ppo_params": {
                "hidden_layers": [128, 64],
                "actor_learning_rate": 3.0e-4,
                "critic_learning_rate": 1.0e-3,
                "gamma": 0.99,
                "lambda_gae": 0.95,
                "clip_ratio": 0.2,
                "target_kl": 0.01,
                "max_grad_norm": 0.5,
                "update_epochs": 10,
                "batch_size": 32,
                "action_std_init": 0.6,
                "action_std_decay_rate": 0.05,
                "min_action_std": 0.1,
                "state_normalize": True
            }
        },
        "rebmbo_deep": {
            "name": "REBMBO-D",
            "type": "rebmbo",
            "variant": "deep",
            "gp_params": {
                "kernel": "deep",
                "deep_network": {
                    "hidden_layers": [128, 64, 32],
                    "activation": "relu",
                    "dropout_rate": 0.1,
                    "weight_decay": 1.0e-4,
                    "batch_size": 64,
                    "epochs": 50,
                    "learning_rate": 0.001
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10,
                "alpha": 1.0e-10
            },
            "ebm_params": {
                "hidden_layers": [128, 64, 32],
                "learning_rate": 0.001,
                "mcmc_steps": 30,
                "mcmc_step_size": 0.1,
                "mcmc_noise_scale": 0.01,
                "lambda": 0.15,
                "update_frequency": 3,
                "batch_size": 64,
                "epochs_per_update": 15,
                "normalize_energy": True
            },
            "ppo_params": {
                "hidden_layers": [256, 128, 64],
                "actor_learning_rate": 2.5e-4,
                "critic_learning_rate": 8.0e-4,
                "gamma": 0.99,
                "lambda_gae": 0.95,
                "clip_ratio": 0.15,
                "target_kl": 0.008,
                "max_grad_norm": 0.5,
                "update_epochs": 15,
                "batch_size": 64,
                "action_std_init": 0.5,
                "action_std_decay_rate": 0.03,
                "min_action_std": 0.08,
                "state_normalize": True
            }
        },
        "experiments": {
            "rebmbo_no_ebm": {
                "name": "REBMBO-NoEBM",
                "type": "rebmbo",
                "variant": "classic",
                "use_ebm": False
            },
            "rebmbo_single_step": {
                "name": "REBMBO-SingleStep",
                "type": "rebmbo",
                "variant": "classic",
                "use_ppo": False
            },
            "rebmbo_high_exploration": {
                "name": "REBMBO-HighExplore",
                "type": "rebmbo",
                "variant": "classic",
                "ebm_params": {
                    "lambda": 0.3
                }
            },
            "rebmbo_low_exploration": {
                "name": "REBMBO-LowExplore",
                "type": "rebmbo",
                "variant": "classic",
                "ebm_params": {
                    "lambda": 0.05
                }
            }
        }
    }
    
    with open('./configs/methods/rebmbo.yaml', 'w') as file:
        yaml.dump(rebmbo_config, file, default_flow_style=False)
    print("Created REBMBO config file: ./configs/methods/rebmbo.yaml")

# Convert baseline configurations to YAML
def create_baseline_config():
    # These are the baseline configurations from your first file
    baseline_config = {
        "common": {
            "initial_points": 5,
            "verbose": True,
            "normalize_inputs": True,
            "seed": 42
        },
        "turbo": {
            "name": "TuRBO",
            "type": "baseline",
            "trust_region_params": {
                "length_init": 0.8,
                "length_min": 0.5e-4,
                "length_max": 1.0,
                "success_tolerance": 3,
                "failure_tolerance": 3,
                "success_factor": 1.5,
                "failure_factor": 0.5,
                "restart_strategy": "random"
            },
            "gp_params": {
                "kernel": "matern",
                "kernel_params": {
                    "nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10,
                "optimizer": "L-BFGS-B",
                "n_restarts_optimizer": 5
            },
            "acquisition_params": {
                "type": "ei",
                "xi": 0.01,
                "optimizer": "L-BFGS-B",
                "n_restarts": 10,
                "acq_optimizer_kwargs": {
                    "samples": 1000,
                    "maxiter": 200
                }
            }
        },
        "ballet_ici": {
            "name": "BALLET-ICI",
            "type": "baseline",
            "global_gp_params": {
                "kernel": "rbf",
                "kernel_params": {
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 5
            },
            "local_gp_params": {
                "kernel": "matern",
                "kernel_params": {
                    "nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 5
            },
            "roi_params": {
                "threshold_method": "quantile",
                "threshold_quantile": 0.9,
                "n_samples": 10000,
                "min_roi_points": 5,
                "inheritance_factor": 0.8
            },
            "acquisition_params": {
                "type": "ucb_lcb_diff",
                "kappa": 2.0,
                "optimizer": "L-BFGS-B",
                "n_restarts": 10,
                "acq_optimizer_kwargs": {
                    "samples": 1000,
                    "maxiter": 200
                }
            }
        },
        "earl_bo": {
            "name": "EARL-BO",
            "type": "baseline",
            "gp_params": {
                "kernel": "rbf_white",
                "kernel_params": {
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10
            },
            "rl_params": {
                "state_dim": 3,
                "hidden_layers": [64, 32],
                "learning_rate": 1.0e-3,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 200,
                "update_frequency": 1,
                "batch_size": 32,
                "memory_size": 10000,
                "target_update": 10,
                "double_dqn": True,
                "dueling_dqn": True,
                "lookahead": 1
            }
        },
        "classic_bo": {
            "name": "Classic BO",
            "type": "baseline",
            "gp_params": {
                "kernel": "matern",
                "kernel_params": {
                    "nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10
            },
            "acquisition_params": {
                "type": "ei",
                "xi": 0.01,
                "optimizer": "L-BFGS-B",
                "n_restarts": 10,
                "acq_optimizer_kwargs": {
                    "samples": 1000,
                    "maxiter": 200
                }
            }
        },
        "two_step_ei": {
            "name": "Two-Step EI",
            "type": "baseline",
            "gp_params": {
                "kernel": "matern",
                "kernel_params": {
                    "nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10
            },
            "acquisition_params": {
                "type": "two_step_ei",
                "xi": 0.01,
                "mc_samples": 100,
                "inner_optimizer": "L-BFGS-B",
                "inner_n_restarts": 5,
                "outer_optimizer": "L-BFGS-B",
                "outer_n_restarts": 10,
                "acq_optimizer_kwargs": {
                    "samples": 1000,
                    "maxiter": 200
                }
            }
        },
        "kg": {
            "name": "Knowledge Gradient",
            "type": "baseline",
            "gp_params": {
                "kernel": "matern",
                "kernel_params": {
                    "nu": 2.5,
                    "length_scale_bounds": [1e-5, 1e5],
                    "output_scale_bounds": [1e-5, 1e5]
                },
                "noise_variance": 1.0e-6,
                "normalize_y": True,
                "optimization_restarts": 10
            },
            "acquisition_params": {
                "type": "kg",
                "mc_samples": 100,
                "inner_samples": 10,
                "optimizer": "L-BFGS-B",
                "n_restarts": 10,
                "acq_optimizer_kwargs": {
                    "samples": 1000,
                    "maxiter": 200
                }
            }
        }
    }
    
    with open('./configs/methods/baselines.yaml', 'w') as file:
        yaml.dump(baseline_config, file, default_flow_style=False)
    print("Created baselines config file: ./configs/methods/baselines.yaml")

def main():
    create_directories()
    create_rebmbo_config()
    create_baseline_config()
    print("Configuration setup complete!")

if __name__ == "__main__":
    main()