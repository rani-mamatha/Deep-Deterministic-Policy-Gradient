"""
Actor Network for DDPGA-TS
Uses Conv1D + GRU + Attention for feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head attention layer"""
    
    def __init__(self, input_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        attn_output, _ = self.attention(x, x, x)
        return attn_output


class ActorNetwork(nn.Module):
    """
    Actor network that outputs continuous actions
    Action: [bandwidth_allocation, cpu_cycles, edge_or_cloud]
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 128], 
                 use_conv=True, use_gru=True, use_attention=True):
        super(ActorNetwork, self).__init__()
        
        self.use_conv = use_conv
        self.use_gru = use_gru
        self.use_attention = use_attention
        
        current_dim = state_dim
        
        # Conv1D for local feature extraction
        if self.use_conv:
            self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(64)
            # After conv, we flatten, so adjust dimension
            current_dim = 64 * state_dim
        
        # GRU for temporal dependencies
        if self.use_gru:
            self.gru_hidden_size = 64
            self.gru = nn.GRU(input_size=current_dim if not use_conv else 64,
                             hidden_size=self.gru_hidden_size, 
                             num_layers=2, 
                             batch_first=True)
            current_dim = self.gru_hidden_size
        
        # Attention mechanism
        if self.use_attention:
            self.attention = AttentionLayer(current_dim, num_heads=4)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim
        
        # Output layer
        self.output = nn.Linear(current_dim, action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
    
    def forward(self, state):
        """
        Forward pass
        state: (batch_size, state_dim)
        """
        x = state
        
        # Conv1D processing
        if self.use_conv:
            # Reshape for Conv1D: (batch, channels=1, length=state_dim)
            x = x.unsqueeze(1)
            x = self.conv1d(x)
            x = self.bn1(x)
            x = F.relu(x)
            # Reshape for GRU: (batch, seq_len=state_dim, features=64)
            x = x.transpose(1, 2)
        else:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # GRU processing
        if self.use_gru:
            x, _ = self.gru(x)
            # Take the last output
            x = x[:, -1, :]
        else:
            x = x.squeeze(1)
        
        # Attention (if GRU is used, expand for attention)
        if self.use_attention and self.use_gru:
            x = x.unsqueeze(1)
            x = self.attention(x)
            x = x.squeeze(1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        
        # Output action with appropriate activations
        action = self.output(x)
        
        # Apply activations:
        # bandwidth and cpu_cycles: sigmoid (0-1 range, will be scaled)
        # edge_or_cloud: sigmoid (interpreted as probability)
        action[:, 0] = torch.sigmoid(action[:, 0])  # bandwidth
        action[:, 1] = torch.sigmoid(action[:, 1])  # cpu_cycles
        action[:, 2] = torch.sigmoid(action[:, 2])  # edge (0) or cloud (1)
        
        return action