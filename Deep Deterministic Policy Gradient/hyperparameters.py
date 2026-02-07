"""
Hyperparameters for DDPGA-TS training
"""

class Hyperparameters:
    """Hyperparameters for Deep Deterministic Policy Gradient algorithms"""
    
    # Learning rates
    ACTOR_LR = 0.0001
    CRITIC_LR = 0.001
    
    # Discount factor
    GAMMA = 0.99
    
    # Soft update parameter (τ)
    TAU = 0.001
    
    # Training parameters
    BATCH_SIZE = 128
    BUFFER_SIZE = 100000
    
    # Episodes and iterations
    MAX_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 100
    
    # Exploration noise parameters (Ornstein-Uhlenbeck)
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    
    # Network architecture
    ACTOR_HIDDEN_DIMS = [256, 128]
    CRITIC_HIDDEN_DIMS = [256, 128]
    
    # GRU parameters
    GRU_HIDDEN_SIZE = 64
    GRU_NUM_LAYERS = 2
    
    # Attention parameters
    ATTENTION_HEADS = 4
    
    # Conv1D parameters
    CONV_FILTERS = 64
    CONV_KERNEL_SIZE = 3
    
    # Pruning parameters
    PRUNING_THRESHOLD = 0.7  # Utility threshold for pruning
    MIN_AVAILABLE_RESOURCES = 3  # Minimum resources to keep available
    
    # Warm-up phase
    WARMUP_EPISODES = 50
    
    # Evaluation
    EVAL_FREQUENCY = 50  # Evaluate every N episodes
    SAVE_FREQUENCY = 100  # Save model every N episodes