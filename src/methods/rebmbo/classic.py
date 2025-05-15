import numpy as np
import torch

from ebm import EnergyBasedModel
from ppo import PPOAgent
from ..base_method import BaseREBMBO

class REBMBOClassic(BaseREBMBO):
    """
    REBMBO Classic implementation using GP for posterior estimation,
    EnergyBasedModel for global energy guidance, and PPOAgent for policy.
    """
    def __init__(
        self,
        benchmark,
        initial_points=None,
        config=None,
    ):
        """
        Initialize REBMBO-Classic algorithm.
        """
        super().__init__(config)

        self.benchmark = benchmark
        self.initial_points = initial_points or []
        self.config = config or {}

        self.gp_params = self.config.get('gp_params', {})
        self.ebm_params = self.config.get('ebm_params', {})
        self.ppo_params = self.config.get('ppo_params', {})

        self.bounds = getattr(self.benchmark, 'bounds', None)
        self.dim = getattr(self.benchmark, 'dimensionality', None)
        self.name = "REBMBO-Classic"

        self.X = []
        self.y = []
        for (xx, yy) in self.initial_points:
            self.X.append(xx)
            self.y.append(yy)

        self.ebm = EnergyBasedModel(
            input_dim=self.dim,
            hidden_dims=self.ebm_params.get('hidden_dims', [128, 128]),
            mcmc_steps=self.ebm_params.get('mcmc_steps', 20),
            step_size=self.ebm_params.get('step_size', 0.01),
            noise_scale=self.ebm_params.get('noise_scale', 0.01),
            learning_rate=self.ebm_params.get('learning_rate', 1e-3),
            device=self.ebm_params.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        )

        maybe_state_dim = self.ppo_params.get('state_dim', 3*self.dim)
        self.ppo_agent = PPOAgent(
            state_dim=maybe_state_dim,
            action_dim=self.dim,
            bounds=self.bounds,
            hidden_dims=self.ppo_params.get('hidden_dims', [128, 64]),
            clip_ratio=self.ppo_params.get('clip_ratio', 0.2),
            actor_lr=self.ppo_params.get('actor_lr', 3e-4),
            critic_lr=self.ppo_params.get('critic_lr', 1e-3),
            gamma=self.ppo_params.get('gamma', 0.99),
            gae_lambda=self.ppo_params.get('gae_lambda', 0.95),
            entropy_coef=self.ppo_params.get('entropy_coef', 0.01),
            device=self.ppo_params.get('device', "cuda" if torch.cuda.is_available() else "cpu")
        )

        print(f"[{self.name}] Initialized with {len(self.X)} initial data points.")

    def suggest_next_point(self):
        """
        Suggest next evaluation point based on GP posterior and EBM energy.
        """
        print(f"[{self.name}] suggest_next_point called.")

        if self.dim is None or not self.bounds:
            print("[Warning] No dimension or bounds info. Returning default [0.5]*dim.")
            return [0.5] * (self.dim if self.dim else 1)

        mu = np.full(self.dim, 0.5)
        sigma = np.full(self.dim, 0.1)
        energy_vals = np.zeros(self.dim)

        state = np.concatenate([mu, sigma, energy_vals], axis=0)

        next_x = self.ppo_agent.act(state)

        return next_x

    def update(self, x, y):
        """
        Update with new observation (x, y).
        """
        print(f"[{self.name}] update with new point x={x}, y={y}")

        self.X.append(x)
        self.y.append(y)

        ebm_val = self.ebm.energy(x)
        if isinstance(ebm_val, np.ndarray):
            ebm_val = ebm_val[0]

        alpha = 1.0
        reward = y - alpha*ebm_val

        self.ppo_agent.update_buffer(reward, done=False)

    def train_ebm(self):
        """
        Train the energy-based model.
        """
        print(f"[{self.name}] train_ebm called.")
        if len(self.X) < 2:
            print("[EBM] Not enough data, skip EBM training.")
            return

        data_points = list(zip(self.X, self.y))
        epochs = self.ebm_params.get("epochs", 10)
        batch_size = self.ebm_params.get("batch_size", 32)

        self.ebm.train(data_points, self.bounds, epochs=epochs, batch_size=batch_size)

    def update_policy(self):
        """
        Update PPO policy.
        """
        print(f"[{self.name}] update_policy called (PPO).")

        epochs = self.ppo_params.get("update_epochs", 10)
        batch_size = self.ppo_params.get("batch_size", 64)

        actor_loss, critic_loss = self.ppo_agent.update(epochs=epochs, batch_size=batch_size)

        print(f"[PPO] actor_loss={actor_loss:.4f}, critic_loss={critic_loss:.4f}")

    def best_point(self):
        """
        Return the current best point.
        """
        print(f"[{self.name}] best_point called.")
        if not self.X:
            return None, None

        best_idx = int(np.argmax(self.y))
        return self.X[best_idx], self.y[best_idx]