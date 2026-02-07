"""
Experience Replay Buffer for DDPG
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""
    
    def __init__(self, buffer_size):
        """
        Initialize replay buffer
        
        Args:
            buffer_size: Maximum size of buffer
        """
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from buffer"""
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = np.array([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences])
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of internal buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()