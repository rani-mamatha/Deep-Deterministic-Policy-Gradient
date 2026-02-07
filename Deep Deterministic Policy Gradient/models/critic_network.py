"""
Critic Network for DDPGA-TS
Q-value function Q(s, a)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """
    Critic network that estimates Q-value Q(s, a)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128]):
        super(CriticNetwork, self).__init__()
        
        # State processing path
        self.state_fc = nn.Linear(state_dim, hidden_dims[0] // 2)
        
        # Action processing path
        self.action_fc = nn.Linear(action_dim, hidden_dims[0] // 2)
        
        # Combined processing
        self.fc_layers = nn.ModuleList()
        current_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            self.fc_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Output Q-value
        self.q_value = nn.Linear(current_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state, action):
        """
        Forward pass
        state: (batch_size, state_dim)
        action: (batch_size, action_dim)
        """
        # Process state and action separately
        state_features = F.relu(self.state_fc(state))
        action_features = F.relu(self.action_fc(action))
        
        # Concatenate
        x = torch.cat([state_features, action_features], dim=1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output Q-value
        q = self.q_value(x)
        
        return q