"""
Ornstein-Uhlenbeck Noise for exploration
"""

import numpy as np


class OrnsteinUhlenbeckNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise
    Used for exploration in continuous action spaces
    """
    
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        """
        Initialize OU Noise
        
        Args:
            action_dim: Dimension of action space
            mu: Mean of the noise
            theta: Rate of mean reversion
            sigma: Volatility parameter
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """Reset the internal state to mean"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state