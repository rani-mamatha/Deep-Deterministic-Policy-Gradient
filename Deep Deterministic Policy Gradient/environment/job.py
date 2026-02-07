"""
Job class representing user tasks
"""

import numpy as np
from config.config import Config


class Job:
    """Represents a user job/task"""
    
    def __init__(self, job_id, data_size=None, cpu_cycles=None, deadline=None):
        self.job_id = job_id
        
        # Job parameters (random if not specified)
        self.data_size = data_size if data_size is not None else \
            np.random.uniform(Config.MIN_DATA_SIZE, Config.MAX_DATA_SIZE)
        
        self.cpu_cycles = cpu_cycles if cpu_cycles is not None else \
            np.random.uniform(Config.MIN_CPU_CYCLES, Config.MAX_CPU_CYCLES)
        
        self.deadline = deadline if deadline is not None else \
            np.random.uniform(Config.MIN_DEADLINE, Config.MAX_DEADLINE)
        
        # Scheduling information
        self.assigned_location = None  # 'edge' or 'cloud'
        self.assigned_vm = None
        self.assigned_bandwidth = None
        self.completion_time = None
        self.is_completed = False
        self.is_rejected = False
    
    def get_tuple(self):
        """Return job as tuple (data_size, cpu_cycles, deadline)"""
        return (self.data_size, self.cpu_cycles, self.deadline)
    
    def __repr__(self):
        return (f"Job(id={self.job_id}, data={self.data_size:.2e}, "
                f"cpu={self.cpu_cycles:.2e}, deadline={self.deadline:.3f})")