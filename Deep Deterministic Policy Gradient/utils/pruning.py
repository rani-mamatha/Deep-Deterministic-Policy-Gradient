"""
Action Space Pruning Strategy (Novel contribution of DDPGA-TS)
"""

import numpy as np
from config.config import Config
from config.hyperparameters import Hyperparameters


class ActionSpacePruner:
    """
    Prunes action space by removing low-utility resources
    This is the novel contribution of the proposed DDPGA-TS algorithm
    """
    
    def __init__(self, resource_manager):
        """
        Initialize pruner
        
        Args:
            resource_manager: ResourceManager instance
        """
        self.resource_manager = resource_manager
        self.pruning_threshold = Hyperparameters.PRUNING_THRESHOLD
        self.min_resources = Hyperparameters.MIN_AVAILABLE_RESOURCES
    
    def prune_vms(self, location='edge'):
        """
        Prune VMs based on utilization
        Returns list of available VMs after pruning
        """
        # Get all VMs for the location
        if location == 'edge':
            vms = self.resource_manager.edge_vms
        else:
            vms = self.resource_manager.cloud_vms
        
        # Calculate utilization ratios
        vm_utils = []
        for vm in vms:
            util_ratio = vm.current_utilization / vm.capacity if vm.capacity > 0 else 1.0
            vm_utils.append((vm, util_ratio))
        
        # Sort by utilization (lower utilization = more available)
        vm_utils.sort(key=lambda x: x[1])
        
        # Keep VMs with low utilization (more available capacity)
        # and ensure minimum number of resources
        pruned_vms = []
        for vm, util in vm_utils:
            # Keep if utilization is below threshold (i.e., has available capacity)
            if util < self.pruning_threshold:
                pruned_vms.append(vm)
            # Or if we haven't met minimum requirement
            elif len(pruned_vms) < self.min_resources:
                pruned_vms.append(vm)
        
        # Ensure at least min_resources are available
        if len(pruned_vms) < self.min_resources:
            pruned_vms = [vm for vm, _ in vm_utils[:self.min_resources]]
        
        return pruned_vms
    
    def prune_channels(self, node_id):
        """
        Prune wireless channels based on utilization
        Returns list of available channel IDs after pruning
        """
        if node_id >= self.resource_manager.num_edge_nodes:
            return []
        
        # Get channel utilizations
        channel_utils = []
        for ch_id in range(self.resource_manager.channels_per_node):
            util = self.resource_manager.channel_utilization[node_id, ch_id]
            channel_utils.append((ch_id, util))
        
        # Sort by utilization (lower = more available)
        channel_utils.sort(key=lambda x: x[1])
        
        # Keep channels with low utilization
        pruned_channels = []
        for ch_id, util in channel_utils:
            if util < self.pruning_threshold:
                pruned_channels.append(ch_id)
            elif len(pruned_channels) < self.min_resources:
                pruned_channels.append(ch_id)
        
        # Ensure minimum channels
        if len(pruned_channels) < self.min_resources:
            pruned_channels = [ch_id for ch_id, _ in channel_utils[:self.min_resources]]
        
        return pruned_channels
    
    def get_pruned_action_space_size(self, location='edge'):
        """
        Get the size of pruned action space
        Returns: (num_vms, num_channels)
        """
        pruned_vms = self.prune_vms(location)
        
        # For channels, average across edge nodes
        if location == 'edge':
            avg_channels = 0
            for node_id in range(self.resource_manager.num_edge_nodes):
                channels = self.prune_channels(node_id)
                avg_channels += len(channels)
            avg_channels = avg_channels / self.resource_manager.num_edge_nodes \
                if self.resource_manager.num_edge_nodes > 0 else 0
        else:
            avg_channels = 0  # Cloud doesn't use wireless channels
        
        return len(pruned_vms), avg_channels
    
    def select_best_vm(self, location='edge', cpu_required=0):
        """
        Select best VM from pruned action space
        Prioritizes VMs with lower utilization for load balancing
        """
        pruned_vms = self.prune_vms(location)
        
        # Filter by capacity requirement
        suitable_vms = [vm for vm in pruned_vms 
                       if vm.get_available_capacity() >= cpu_required]
        
        if not suitable_vms:
            return None
        
        # Select VM with most available capacity (for load balancing)
        best_vm = max(suitable_vms, key=lambda vm: vm.get_available_capacity())
        return best_vm
    
    def select_best_channel(self, node_id):
        """
        Select best wireless channel from pruned action space
        Prioritizes channels with lower utilization
        """
        pruned_channels = self.prune_channels(node_id)
        
        if not pruned_channels:
            return None
        
        # Select channel with lowest utilization
        best_channel = min(pruned_channels, 
                          key=lambda ch: self.resource_manager.channel_utilization[node_id, ch])
        return best_channel
    
    def get_pruning_stats(self):
        """Get statistics about pruning effectiveness"""
        # Edge VMs
        total_edge_vms = len(self.resource_manager.edge_vms)
        pruned_edge_vms = len(self.prune_vms('edge'))
        edge_reduction = (total_edge_vms - pruned_edge_vms) / total_edge_vms \
            if total_edge_vms > 0 else 0
        
        # Cloud VMs
        total_cloud_vms = len(self.resource_manager.cloud_vms)
        pruned_cloud_vms = len(self.prune_vms('cloud'))
        cloud_reduction = (total_cloud_vms - pruned_cloud_vms) / total_cloud_vms \
            if total_cloud_vms > 0 else 0
        
        # Channels (average across nodes)
        total_channels = self.resource_manager.channels_per_node
        avg_pruned_channels = 0
        for node_id in range(self.resource_manager.num_edge_nodes):
            avg_pruned_channels += len(self.prune_channels(node_id))
        avg_pruned_channels = avg_pruned