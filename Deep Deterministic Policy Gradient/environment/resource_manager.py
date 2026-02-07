"""
Resource Manager for Edge-Cloud Environment
"""

import numpy as np
from config.config import Config


class VM:
    """Virtual Machine"""
    
    def __init__(self, vm_id, location, capacity):
        self.vm_id = vm_id
        self.location = location  # 'edge' or 'cloud'
        self.capacity = capacity  # CPU cores
        self.current_utilization = 0.0
        self.jobs_running = []
    
    def is_available(self):
        """Check if VM has available resources"""
        return self.current_utilization < self.capacity
    
    def get_available_capacity(self):
        """Get remaining capacity"""
        return self.capacity - self.current_utilization
    
    def allocate(self, cpu_required):
        """Allocate resources"""
        if self.current_utilization + cpu_required <= self.capacity:
            self.current_utilization += cpu_required
            return True
        return False
    
    def release(self, cpu_amount):
        """Release resources"""
        self.current_utilization = max(0, self.current_utilization - cpu_amount)


class ResourceManager:
    """Manages resources in edge and cloud"""
    
    def __init__(self, num_edge_nodes, num_cloud_nodes):
        self.num_edge_nodes = num_edge_nodes
        self.num_cloud_nodes = num_cloud_nodes
        
        # Initialize VMs
        self.edge_vms = []
        self.cloud_vms = []
        
        # Create edge VMs
        vm_id = 0
        for node_id in range(num_edge_nodes):
            for _ in range(Config.VMS_PER_EDGE_NODE):
                vm = VM(vm_id, 'edge', Config.VM_CAPACITY_EDGE)
                self.edge_vms.append(vm)
                vm_id += 1
        
        # Create cloud VMs
        for server_id in range(num_cloud_nodes):
            for _ in range(Config.VMS_PER_CLOUD_SERVER):
                vm = VM(vm_id, 'cloud', Config.VM_CAPACITY_CLOUD)
                self.cloud_vms.append(vm)
                vm_id += 1
        
        # Wireless channels for edge nodes
        self.channels_per_node = Config.CHANNELS_PER_NODE
        self.channel_bandwidth = Config.BANDWIDTH_PER_CHANNEL
        self.channel_utilization = np.zeros((num_edge_nodes, self.channels_per_node))
    
    def get_available_vms(self, location='edge', min_capacity=0):
        """Get list of available VMs at specified location"""
        if location == 'edge':
            vms = self.edge_vms
        else:
            vms = self.cloud_vms
        
        available = [vm for vm in vms if vm.get_available_capacity() >= min_capacity]
        return available
    
    def get_available_channels(self, node_id):
        """Get available wireless channels for edge node"""
        if node_id >= self.num_edge_nodes:
            return []
        
        available = []
        for ch_id in range(self.channels_per_node):
            if self.channel_utilization[node_id, ch_id] < 1.0:
                available.append(ch_id)
        return available
    
    def allocate_channel(self, node_id, channel_id, bandwidth_ratio):
        """Allocate wireless channel bandwidth"""
        if node_id < self.num_edge_nodes and channel_id < self.channels_per_node:
            if self.channel_utilization[node_id, channel_id] + bandwidth_ratio <= 1.0:
                self.channel_utilization[node_id, channel_id] += bandwidth_ratio
                return True
        return False
    
    def release_channel(self, node_id, channel_id, bandwidth_ratio):
        """Release wireless channel bandwidth"""
        if node_id < self.num_edge_nodes and channel_id < self.channels_per_node:
            self.channel_utilization[node_id, channel_id] = max(
                0, self.channel_utilization[node_id, channel_id] - bandwidth_ratio
            )
    
    def get_total_utilization(self):
        """Get total resource utilization across all VMs"""
        total_capacity = 0
        total_utilization = 0
        
        for vm in self.edge_vms + self.cloud_vms:
            total_capacity += vm.capacity
            total_utilization += vm.current_utilization
        
        return total_utilization / total_capacity if total_capacity > 0 else 0
    
    def get_edge_utilization(self):
        """Get edge resource utilization"""
        total_capacity = sum(vm.capacity for vm in self.edge_vms)
        total_utilization = sum(vm.current_utilization for vm in self.edge_vms)
        return total_utilization / total_capacity if total_capacity > 0 else 0
    
    def get_cloud_utilization(self):
        """Get cloud resource utilization"""
        total_capacity = sum(vm.capacity for vm in self.cloud_vms)
        total_utilization = sum(vm.current_utilization for vm in self.cloud_vms)
        return total_utilization / total_capacity if total_capacity > 0 else 0
    
    def reset(self):
        """Reset all resource utilizations"""
        for vm in self.edge_vms + self.cloud_vms:
            vm.current_utilization = 0.0
            vm.jobs_running = []
        
        self.channel_utilization.fill(0)