"""
DDPG Agent - Base agent for all variants
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.actor_network import ActorNetwork
from models.critic_network import CriticNetwork
from utils.replay_buffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeckNoise
from config.hyperparameters import Hyperparameters


class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent"""
    
    def __init__(self, state_dim, action_dim, device='cpu', 
                 use_conv=True, use_gru=True, use_attention=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Hyperparameters
        self.gamma = Hyperparameters.GAMMA
        self.tau = Hyperparameters.TAU
        self.batch_size = Hyperparameters.BATCH_SIZE
        
        # Actor networks (online and target)
        self.actor = ActorNetwork(
            state_dim, action_dim, 
            Hyperparameters.ACTOR_HIDDEN_DIMS,
            use_conv, use_gru, use_attention
        ).to(device)
        
        self.actor_target = ActorNetwork(
            state_dim, action_dim, 
            Hyperparameters.ACTOR_HIDDEN_DIMS,
            use_conv, use_gru, use_attention
        ).to(device)
        
        # Copy parameters to target network
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (online and target)
        self.critic = CriticNetwork(
            state_dim, action_dim, 
            Hyperparameters.CRITIC_HIDDEN_DIMS
        ).to(device)
        
        self.critic_target = CriticNetwork(
            state_dim, action_dim, 
            Hyperparameters.CRITIC_HIDDEN_DIMS
        ).to(device)
        
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=Hyperparameters.ACTOR_LR
        )
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=Hyperparameters.CRITIC_LR
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(Hyperparameters.BUFFER_SIZE)
        
        # Exploration noise
        self.noise = OrnsteinUhlenbeckNoise(
            action_dim, 
            theta=Hyperparameters.NOISE_THETA,
            sigma=Hyperparameters.NOISE_SIGMA
        )
        
        # Training mode flag
        self.training_mode = True
    
    def select_action(self, state, add_noise=True):
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        # Add exploration noise during training
        if add_noise and self.training_mode:
            noise = self.noise.sample()
            action = np.clip(action + noise, 0, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self):
        """Train the agent using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        # Sample batch
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ------------------- Update Critic ------------------- #
        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Compute current Q-value
        current_q = self.critic(states, actions)
        
        # Critic loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ------------------- Update Actor ------------------- #
        # Actor loss (negative Q-value to maximize)
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------- Soft Update Target Networks ------------------- #
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, local_model, target_model):
        """Soft update target network parameters
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), 
                                             local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def reset_noise(self):
        """Reset exploration noise"""
        self.noise.reset()
    
    def save(self, filepath):
        """Save model parameters"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Update target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())