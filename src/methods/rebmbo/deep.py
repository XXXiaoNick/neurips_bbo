from ..base_method import BaseREBMBO

class REBMBODeep(BaseREBMBO):

    def __init__(self, benchmark, initial_points=None, config=None):
        super().__init__(config)

        self.benchmark = benchmark
        self.initial_points = initial_points or []
        self.config = config or {}

        self.gp_params = self.config.get('gp_params', {})
        self.ebm_params = self.config.get('ebm_params', {})
        self.ppo_params = self.config.get('ppo_params', {})

        self.deep_network = self.gp_params.get('deep_network', {})

        self.bounds = getattr(self.benchmark, 'bounds', None)
        self.name = "REBMBO-Deep"

        print(f"Initialized {self.name} with {len(self.initial_points)} initial points.")

    def suggest_next_point(self):
        print(f"[{self.name}] Suggesting the next point using Deep EBM-UCB logic.")
        return [0.5] * len(self.bounds or [])

    def update(self, x, y):
        print(f"[{self.name}] Updating with new point: {x}, value: {y}")

    def train_ebm(self):
        print(f"[{self.name}] Training EBM with MCMC.")

    def update_policy(self):
        print(f"[{self.name}] Updating PPO policy with multi-step RL approach.")

    def best_point(self):
        print(f"[{self.name}] Returning the best point found so far.")
        return [0.5] * len(self.bounds or []), 0.0
