"""
Hyperparameters for DDPGA-TS training
OPTIMIZED FOR CPU EXECUTION
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
    
    # Training parameters - OPTIMIZED FOR CPU
    BATCH_SIZE = 64              # Reduced from 128 for faster CPU processing
    BUFFER_SIZE = 50000          # Reduced from 100000 to save memory
    
    # Episodes and iterations - REDUCED FOR FASTER TESTING
    MAX_EPISODES = 1000          # Keep at 1000 for full results
    # For quick testing, you can reduce to 200-300
    MAX_STEPS_PER_EPISODE = 100
    
    # Exploration noise parameters (Ornstein-Uhlenbeck)
    NOISE_THETA = 0.15
    NOISE_SIGMA = 0.2
    
    # Network architecture - OPTIMIZED FOR CPU
    ACTOR_HIDDEN_DIMS = [128, 64]     # Reduced from [256, 128]
    CRITIC_HIDDEN_DIMS = [128, 64]    # Reduced from [256, 128]
    
    # GRU parameters - OPTIMIZED FOR CPU
    GRU_HIDDEN_SIZE = 32              # Reduced from 64
    GRU_NUM_LAYERS = 1                # Reduced from 2
    
    # Attention parameters - OPTIMIZED FOR CPU
    ATTENTION_HEADS = 2               # Reduced from 4
    
    # Conv1D parameters - OPTIMIZED FOR CPU
    CONV_FILTERS = 32                 # Reduced from 64
    CONV_KERNEL_SIZE = 3
    
    # Pruning parameters
    PRUNING_THRESHOLD = 0.7
    MIN_AVAILABLE_RESOURCES = 3
    
    # Warm-up phase
    WARMUP_EPISODES = 50
    
    # Evaluation
    EVAL_FREQUENCY = 50
    SAVE_FREQUENCY = 100