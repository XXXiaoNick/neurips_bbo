import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import MultivariateNormal

class ActorNetwork(nn.Module):
    """Actor network for generating policy distributions"""
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super(ActorNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        
        self.network = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        
        # Learn action standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        """
        Forward pass, returns mean and std of multivariate Gaussian distribution
        
        Args:
            state: State representation
            
        Returns:
            mean: Mean of action distribution
            std: Standard deviation of action distribution
        """
        x = self.network(state)
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std


class CriticNetwork(nn.Module):
    """Critic network for estimating state values"""
    def __init__(self, state_dim, hidden_dims=[128, 64]):
        super(CriticNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Estimate the value function for a state
        
        Args:
            state: State representation
            
        Returns:
            value: State value function
        """
        return self.network(state)


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Used for multi-step decision making in Bayesian optimization,
    combining information from GP and EBM.
    """
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 bounds,
                 hidden_dims=[128, 64],
                 clip_ratio=0.2,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 entropy_coef=0.01,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize PPO agent
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            bounds: Action space boundaries [(lower_1, upper_1), ..., (lower_d, upper_d)]
            hidden_dims: Hidden layer dimensions
            clip_ratio: PPO clipping ratio
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            gae_lambda: GAE parameter
            entropy_coef: Entropy coefficient
            device: Computation device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.bounds = bounds
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.device = device
        
        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize buffer
        self.reset_buffer()
        
        # Initialize noise and clipping limits
        self.action_std_decay_freq = 50
        self.initial_action_std = 0.6
        self.min_action_std = 0.1
        self.action_std = self.initial_action_std
        
    def reset_buffer(self):
        """Reset experience replay buffer"""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'terminals': []
        }
        
    def act(self, state):
        """
        Select action based on current state
        
        Args:
            state: Current state representation
            
        Returns:
            action: Selected action
        """
        # Ensure state is a tensor
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            self.actor.eval()
            self.critic.eval()
            
            # Get action distribution
            mean, std = self.actor(state)
            
            # Sample action using multivariate Gaussian distribution
            cov_matrix = torch.diag(std.squeeze() ** 2)
            dist = MultivariateNormal(mean, cov_matrix)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state)
            
            # Convert action back to original range
            action_np = action.cpu().numpy().squeeze()
            
            # Map action to boundary range
            scaled_action = np.zeros_like(action_np)
            for i, (lower, upper) in enumerate(self.bounds):
                # Map from standard Gaussian space to actual boundaries
                scaled_action[i] = lower + (upper - lower) * (action_np[i] + 5) / 10.0
                # Ensure it doesn't exceed boundaries
                scaled_action[i] = np.clip(scaled_action[i], lower, upper)
            
            # Save to buffer
            self.buffer['states'].append(state.cpu().numpy().squeeze())
            self.buffer['actions'].append(action_np)
            self.buffer['log_probs'].append(log_prob.cpu().item())
            self.buffer['values'].append(value.cpu().item())
            
            return scaled_action
        
    def update_buffer(self, reward, done=False):
        """
        Update buffer
        
        Args:
            reward: Received reward
            done: Whether the episode is done
        """
        self.buffer['rewards'].append(reward)
        self.buffer['terminals'].append(done)
        
    def update(self, epochs=10, batch_size=64):
        """
        Update policy using PPO algorithm
        
        Args:
            epochs: Number of update epochs
            batch_size: Batch size
            
        Returns:
            actor_loss: Actor loss
            critic_loss: Critic loss
        """
        # Check if buffer has enough data
        if len(self.buffer['rewards']) < 2:
            return 0, 0
            
        # Prepare data
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        old_values = np.array(self.buffer['values'])
        old_log_probs = np.array(self.buffer['log_probs'])
        terminals = np.array(self.buffer['terminals'])
        
        # Calculate Generalized Advantage Estimation (GAE)
        advantages = self._compute_advantages(rewards, old_values, terminals)
        
        # Convert data to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(old_values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Set to training mode
        self.actor.train()
        self.critic.train()
        
        # Multiple training epochs
        actor_losses = []
        critic_losses = []
        
        for _ in range(epochs):
            # Use minibatch if data volume is sufficient
            if len(rewards) > batch_size:
                indices = np.arange(len(rewards))
                np.random.shuffle(indices)
                
                for start_idx in range(0, len(rewards), batch_size):
                    end_idx = min(start_idx + batch_size, len(rewards))
                    batch_indices = indices[start_idx:end_idx]
                    
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    actor_loss, critic_loss = self._update_batch(
                        batch_states, batch_actions, batch_old_log_probs, 
                        batch_advantages, batch_returns)
                    
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            else:
                # Use all data if not enough
                actor_loss, critic_loss = self._update_batch(
                    states, actions, old_log_probs, advantages, returns)
                
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
        
        # Clear buffer
        self.reset_buffer()
        
        # Update action standard deviation
        self._decay_action_std()
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    def _update_batch(self, states, actions, old_log_probs, advantages, returns):
        """
        Update networks using a single batch
        
        Args:
            states: State batch
            actions: Action batch
            old_log_probs: Old action log probabilities
            advantages: Advantage estimates
            returns: Return estimates
            
        Returns:
            actor_loss: Actor loss
            critic_loss: Critic loss
        """
        # Calculate current policy action probabilities
        mean, std = self.actor(states)
        cov_matrix = torch.diag_embed(std ** 2)
        dist = MultivariateNormal(mean, cov_matrix)
        
        current_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate ratio and clipped target
        ratios = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # Actor loss
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Critic loss
        current_values = self.critic(states).squeeze()
        value_loss = nn.MSELoss()(current_values, returns)
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), value_loss.item()
    
    def _compute_advantages(self, rewards, values, terminals):
        """
        Calculate Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Reward sequence
            values: Value function estimates
            terminals: Terminal state flags
            
        Returns:
            advantages: Advantage estimates
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]
        
        # Calculate advantages from back to front
        for i in reversed(range(len(rewards))):
            if terminals[i]:
                # No subsequent rewards if terminal state
                delta = rewards[i] - values[i]
                advantages[i] = delta
            else:
                # Calculate TD error
                if i == len(rewards) - 1:
                    next_value = last_value
                else:
                    next_value = values[i + 1]
                    
                delta = rewards[i] + self.gamma * next_value - values[i]
                advantages[i] = delta + self.gamma * self.gae_lambda * last_advantage
                
            last_advantage = advantages[i]
            
        return advantages
    
    def _decay_action_std(self):
        """Decay action standard deviation to reduce exploration"""
        if hasattr(self, 'steps'):
            self.steps += 1
        else:
            self.steps = 1
            
        if self.steps % self.action_std_decay_freq == 0:
            self.action_std = max(self.min_action_std, 
                                 self.action_std - 0.01)
            
            # Update actor's log_std parameter
            with torch.no_grad():
                self.actor.log_std.copy_(torch.log(torch.ones_like(self.actor.log_std) * self.action_std))
    
    def save(self, path):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_std': self.action_std
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        
        # Rebuild networks
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        
        # Load states
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.action_std = checkpoint['action_std']
        
        # Update actor's log_std parameter
        with torch.no_grad():
            self.actor.log_std.copy_(torch.log(torch.ones_like(self.actor.log_std) * self.action_std))