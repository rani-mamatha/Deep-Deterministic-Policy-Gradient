"""
DDPG-NN: DDPG with standard Neural Network
Baseline method without Conv1D, GRU, or Attention
"""

import torch
import numpy as np
from models.ddpg_agent import DDPGAgent


class DDPG_NN:
    """
    DDPG with standard NN (baseline)
    No Conv1D, GRU, or Attention
    """
    
    def __init__(self, env, device='cpu'):
        """
        Initialize DDPG-NN
        
        Args:
            env: EdgeCloudEnvironment instance
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.device = device
        
        # DDPG Agent without advanced features
        self.agent = DDPGAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=device,
            use_conv=False,     # No Conv1D
            use_gru=False,      # No GRU
            use_attention=False # No Attention
        )
        
        self.name = "DDPG-NN"
    
    def train_episode(self):
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        actor_losses = []
        critic_losses = []
        
        done = False
        step = 0
        
        while not done:
            # Select action (no pruning)
            action = self.agent.select_action(state, add_noise=True)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            actor_loss, critic_loss = self.agent.train()
            
            if actor_loss is not None:
                actor_losses.append(actor_loss)
            if critic_loss is not None:
                critic_losses.append(critic_loss)
            
            episode_reward += reward
            state = next_state
            step += 1
        
        # Get environment metrics
        env_metrics = self.env.get_metrics()
        
        # Average losses
        avg_actor_loss = np.mean(actor_losses) if actor_losses else None
        avg_critic_loss = np.mean(critic_losses) if critic_losses else None
        
        return episode_reward, avg_actor_loss, avg_critic_loss, env_metrics
    
    def evaluate(self):
        """Evaluate the agent"""
        self.agent.training_mode = False
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = self.agent.select_action(state, add_noise=False)
            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            state = next_state
        
        env_metrics = self.env.get_metrics()
        self.agent.training_mode = True
        
        return episode_reward, env_metrics
    
    def save(self, filepath):
        """Save model"""
        self.agent.save(filepath)
    
    def load(self, filepath):
        """Load model"""
        self.agent.load(filepath)