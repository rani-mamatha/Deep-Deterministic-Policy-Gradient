"""
Edge-Cloud Environment for Task Scheduling
"""

import numpy as np
from environment.job import Job
from environment.resource_manager import ResourceManager
from config.config import Config


class EdgeCloudEnvironment:
    """Edge-Cloud computing environment for task scheduling"""
    
    def __init__(self, num_edge_nodes, num_cloud_nodes, num_jobs):
        self.num_edge_nodes = num_edge_nodes
        self.num_cloud_nodes = num_cloud_nodes
        self.num_jobs = num_jobs
        
        # Resource manager
        self.resource_manager = ResourceManager(num_edge_nodes, num_cloud_nodes)
        
        # Jobs
        self.jobs = []
        self.current_job_idx = 0
        
        # Metrics
        self.total_reward = 0
        self.completed_jobs = 0
        self.rejected_jobs = 0
        self.total_cost = 0
        self.total_gain = 0
        
        # State dimension
        # State = [job_data, job_cpu, job_deadline, edge_util, cloud_util, total_util]
        self.state_dim = 6
        
        # Action dimension
        # Action = [bandwidth_allocation, cpu_cycles, edge_or_cloud]
        self.action_dim = 3
    
    def reset(self):
        """Reset environment"""
        # Generate jobs
        self.jobs = [Job(i) for i in range(self.num_jobs)]
        self.current_job_idx = 0
        
        # Reset resource manager
        self.resource_manager.reset()
        
        # Reset metrics
        self.total_reward = 0
        self.completed_jobs = 0
        self.rejected_jobs = 0
        self.total_cost = 0
        self.total_gain = 0
        
        # Return initial state
        return self._get_state()
    
    def _get_state(self):
        """Get current state"""
        if self.current_job_idx >= len(self.jobs):
            # No more jobs
            return np.zeros(self.state_dim)
        
        job = self.jobs[self.current_job_idx]
        
        # Normalize job parameters
        data_normalized = job.data_size / Config.MAX_DATA_SIZE
        cpu_normalized = job.cpu_cycles / Config.MAX_CPU_CYCLES
        deadline_normalized = job.deadline / Config.MAX_DEADLINE
        
        # Resource utilization
        edge_util = self.resource_manager.get_edge_utilization()
        cloud_util = self.resource_manager.get_cloud_utilization()
        total_util = self.resource_manager.get_total_utilization()
        
        state = np.array([
            data_normalized,
            cpu_normalized,
            deadline_normalized,
            edge_util,
            cloud_util,
            total_util
        ])
        
        return state
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        if self.current_job_idx >= len(self.jobs):
            return self._get_state(), 0, True, {}
        
        job = self.jobs[self.current_job_idx]
        
        # Parse action
        bandwidth_ratio = action[0]  # 0-1
        cpu_ratio = action[1]  # 0-1
        location_prob = action[2]  # 0-1, >0.5 means cloud
        
        # Determine location
        location = 'cloud' if location_prob > 0.5 else 'edge'
        
        # Scale bandwidth and CPU
        if location == 'edge':
            max_bandwidth = Config.BANDWIDTH_PER_CHANNEL
            bandwidth = bandwidth_ratio * max_bandwidth
            cpu_required = cpu_ratio * Config.VM_CAPACITY_EDGE
        else:
            max_bandwidth = Config.BANDWIDTH_EDGE_CLOUD
            bandwidth = bandwidth_ratio * max_bandwidth
            cpu_required = cpu_ratio * Config.VM_CAPACITY_CLOUD
        
        # Try to schedule job
        success, rtt, cost, gain = self._schedule_job(
            job, location, bandwidth, cpu_required
        )
        
        # Calculate reward
        if success:
            reward = gain - cost
            self.completed_jobs += 1
            job.is_completed = True
        else:
            reward = -1.0  # Penalty for rejection
            self.rejected_jobs += 1
            job.is_rejected = True
        
        self.total_reward += reward
        self.total_cost += cost if success else 0
        self.total_gain += gain if success else 0
        
        # Move to next job
        self.current_job_idx += 1
        
        # Get next state
        next_state = self._get_state()
        
        # Check if done
        done = self.current_job_idx >= len(self.jobs)
        
        # Info
        info = {
            'success': success,
            'rtt': rtt if success else None,
            'cost': cost,
            'gain': gain,
            'location': location
        }
        
        return next_state, reward, done, info
    
    def _schedule_job(self, job, location, bandwidth, cpu_required):
        """
        Schedule job and compute RTT, cost, gain
        Returns: (success, rtt, cost, gain)
        """
        # Check resource availability
        available_vms = self.resource_manager.get_available_vms(
            location, cpu_required
        )
        
        if not available_vms:
            return False, 0, 0, 0
        
        # Calculate RTT (Round Trip Time) - Equations 9 and 10
        if location == 'edge':
            propagation_time = Config.PROPAGATION_TIME_EDGE
            transmission_time = job.data_size / bandwidth if bandwidth > 0 else float('inf')
            processing_time = job.cpu_cycles / (cpu_required * Config.CPU_FREQ_EDGE) \
                if cpu_required > 0 else float('inf')
            rtt = 2 * (propagation_time + transmission_time + processing_time)
        else:
            propagation_time = Config.PROPAGATION_TIME_CLOUD
            transmission_time = job.data_size / bandwidth if bandwidth > 0 else float('inf')
            processing_time = job.cpu_cycles / (cpu_required * Config.CPU_FREQ_CLOUD) \
                if cpu_required > 0 else float('inf')
            rtt = 2 * (propagation_time + transmission_time + processing_time)
        
        # Check deadline
        if rtt > job.deadline:
            return False, 0, 0, 0
        
        # Calculate gain (Equation 2)
        gain = Config.SERVICE_GAIN * (job.deadline - rtt)
        
        # Calculate cost (Equation 3)
        if location == 'edge':
            cost_bw = Config.COST_BANDWIDTH_EDGE * bandwidth * rtt
            cost_vm = Config.COST_VM_EDGE * cpu_required * rtt
        else:
            cost_bw = Config.COST_BANDWIDTH_CLOUD * bandwidth * rtt
            cost_vm = Config.COST_VM_CLOUD * cpu_required * rtt
        
        cost = cost_bw + cost_vm
        
        # Allocate resources (simplified - just pick first available VM)
        vm = available_vms[0]
        if vm.allocate(cpu_required):
            job.assigned_location = location
            job.assigned_vm = vm.vm_id
            job.assigned_bandwidth = bandwidth
            job.completion_time = rtt
            return True, rtt, cost, gain
        
        return False, 0, 0, 0
    
    def get_metrics(self):
        """Get environment metrics"""
        acceptance_rate = self.completed_jobs / self.current_job_idx \
            if self.current_job_idx > 0 else 0
        rejection_rate = self.rejected_jobs / self.current_job_idx \
            if self.current_job_idx > 0 else 0
        avg_cost = self.total_cost / self.completed_jobs \
            if self.completed_jobs > 0 else 0
        qoe = self.total_gain / (self.total_cost + 1e-6)  # Quality of Experience
        
        return {
            'total_reward': self.total_reward,
            'completed_jobs': self.completed_jobs,
            'rejected_jobs': self.rejected_jobs,
            'acceptance_rate': acceptance_rate,
            'rejection_rate': rejection_rate,
            'avg_cost': avg_cost,
            'total_cost': self.total_cost,
            'total_gain': self.total_gain,
            'qoe': qoe
        }