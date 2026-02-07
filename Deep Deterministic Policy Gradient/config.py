"""
Configuration file for DDPGA-TS Edge-Cloud Task Scheduling
"""

class Config:
    """Configuration parameters for the edge-cloud environment"""
    
    # Environment scales
    SMALL_SCALE = {
        'edge_nodes': 10,
        'cloud_nodes': 10,
        'num_jobs': 10000,
        'name': 'small'
    }
    
    MEDIUM_SCALE = {
        'edge_nodes': 30,
        'cloud_nodes': 30,
        'num_jobs': 100000,
        'name': 'medium'
    }
    
    LARGE_SCALE = {
        'edge_nodes': 50,
        'cloud_nodes': 50,
        'num_jobs': 1000000,
        'name': 'large'
    }
    
    # Network parameters
    BANDWIDTH_EDGE_CLOUD = 1e9  # 1 Gbps in bits per second
    PROPAGATION_TIME_EDGE = 0.005  # 5ms
    PROPAGATION_TIME_CLOUD = 0.050  # 50ms
    
    # Resource parameters
    VM_CAPACITY_EDGE = 4  # CPU cores per edge VM
    VM_CAPACITY_CLOUD = 8  # CPU cores per cloud VM
    VMS_PER_EDGE_NODE = 5
    VMS_PER_CLOUD_SERVER = 10
    
    # Wireless channels per edge node
    CHANNELS_PER_NODE = 10
    BANDWIDTH_PER_CHANNEL = 100e6  # 100 Mbps per channel
    
    # Cost parameters
    COST_BANDWIDTH_EDGE = 0.001  # ch - cost per unit bandwidth per unit time (edge)
    COST_BANDWIDTH_CLOUD = 0.0005  # cost per unit bandwidth per unit time (cloud)
    COST_VM_EDGE = 0.01  # ck - cost per VM per unit time (edge)
    COST_VM_CLOUD = 0.005  # cost per VM per unit time (cloud)
    
    # Gain parameter
    SERVICE_GAIN = 1.0  # δ - service provider gain
    
    # Job parameters
    MIN_DATA_SIZE = 1e6  # 1 MB
    MAX_DATA_SIZE = 100e6  # 100 MB
    MIN_CPU_CYCLES = 1e9  # 1 GHz-seconds
    MAX_CPU_CYCLES = 10e9  # 10 GHz-seconds
    MIN_DEADLINE = 0.1  # seconds
    MAX_DEADLINE = 5.0  # seconds
    
    # CPU frequency (cycles per second)
    CPU_FREQ_EDGE = 2.5e9  # 2.5 GHz
    CPU_FREQ_CLOUD = 3.5e9  # 3.5 GHz
    
    @staticmethod
    def get_environment_config(scale='small'):
        """Get environment configuration for specified scale"""
        if scale == 'small':
            return Config.SMALL_SCALE
        elif scale == 'medium':
            return Config.MEDIUM_SCALE
        elif scale == 'large':
            return Config.LARGE_SCALE
        else:
            raise ValueError(f"Unknown scale: {scale}")