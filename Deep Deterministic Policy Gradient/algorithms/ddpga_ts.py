"""
DDPGA-TS: Deep Deterministic Policy Gradient Algorithm for Dynamic Task Scheduling
This is the PROPOSED algorithm with pruning strategy
"""

import torch
import numpy as np
from models.ddpg_agent import DDPGAgent
from utils.pruning import ActionSpacePruner
from environment.edge_cloud_env import EdgeCloudEnvironment


class DDPGA_TS:
    """
    Proposed DDPGA-TS Algorithm
    Features:
    - Conv1D for local feature extraction
    - GRU for temporal dependencies
    - Attention mechanism
    - Novel pruning strategy for action space reduction
    """
    
    def __init__(self, env, device='cpu'):
        """
        Initialize DDPGA-TS
        
        Args:
            env: EdgeCloudEnvironment instance
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.device = device
        
        # DDPG Agent with full architecture (Conv1D + GRU + Attention)
        self.agent = DDPGAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            device=device,
            use_conv=True,      # Use Conv1D
            use_gru=True,       # Use GRU
            use_attention=True  # Use Attention
        )
        
        # Action space pruner (Novel contribution)
        self.pruner = ActionSpacePruner(env.resource_manager)
        
        self.name = "DDPGA-TS (Proposed)"
    
    def train_episode(self):
        """Train for one episode"""
        state = self.env.reset()
        episode_reward = 0
        actor_losses = []
        critic_losses = []
        
        done = False
        step = 0
        
        while not done:
            # Prune action space (Novel contribution)
            self.pruner.prune_vms('edge')
            self.pruner.prune_vms('cloud')
            
            # Select action
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
        """Evaluate the agent (no exploration noise)"""
        self.agent.training_mode = False
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # No noise during evaluation
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