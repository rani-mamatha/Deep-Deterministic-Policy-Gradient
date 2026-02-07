"""
Metrics Tracker for training and evaluation
"""

import numpy as np
import json
import os


class MetricsTracker:
    """Track and store training/evaluation metrics"""
    
    def __init__(self, save_dir):
        """
        Initialize metrics tracker
        
        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_losses = []
        self.actor_losses = []
        self.critic_losses = []
        
        # Environment metrics
        self.acceptance_rates = []
        self.rejection_rates = []
        self.operational_costs = []
        self.qoe_scores = []
        
        # Convergence metrics
        self.normalized_rewards = []
        self.normalized_losses = []
    
    def add_episode(self, episode_num, total_reward, actor_loss, critic_loss, 
                    env_metrics):
        """Add metrics for an episode"""
        self.episode_rewards.append(total_reward)
        
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        
        # Environment metrics
        self.acceptance_rates.append(env_metrics.get('acceptance_rate', 0))
        self.rejection_rates.append(env_metrics.get('rejection_rate', 0))
        self.operational_costs.append(env_metrics.get('avg_cost', 0))
        self.qoe_scores.append(env_metrics.get('qoe', 0))
    
    def get_recent_avg_reward(self, num_episodes=100):
        """Get average reward over recent episodes"""
        if len(self.episode_rewards) == 0:
            return 0
        recent = self.episode_rewards[-num_episodes:]
        return np.mean(recent)
    
    def get_recent_avg_cost(self, num_episodes=100):
        """Get average operational cost over recent episodes"""
        if len(self.operational_costs) == 0:
            return 0
        recent = self.operational_costs[-num_episodes:]
        return np.mean(recent)
    
    def get_recent_avg_rejection_rate(self, num_episodes=100):
        """Get average rejection rate over recent episodes"""
        if len(self.rejection_rates) == 0:
            return 0
        recent = self.rejection_rates[-num_episodes:]
        return np.mean(recent)
    
    def get_recent_avg_qoe(self, num_episodes=100):
        """Get average QoE over recent episodes"""
        if len(self.qoe_scores) == 0:
            return 0
        recent = self.qoe_scores[-num_episodes:]
        return np.mean(recent)
    
    def normalize_rewards(self):
        """Normalize rewards using min-max normalization"""
        if len(self.episode_rewards) == 0:
            return []
        
        rewards = np.array(self.episode_rewards)
        min_reward = rewards.min()
        max_reward = rewards.max()
        
        if max_reward - min_reward == 0:
            return np.zeros_like(rewards)
        
        normalized = (rewards - min_reward) / (max_reward - min_reward)
        self.normalized_rewards = normalized.tolist()
        return normalized
    
    def normalize_losses(self):
        """Normalize critic losses using min-max normalization"""
        if len(self.critic_losses) == 0:
            return []
        
        losses = np.array(self.critic_losses)
        min_loss = losses.min()
        max_loss = losses.max()
        
        if max_loss - min_loss == 0:
            return np.zeros_like(losses)
        
        normalized = (losses - min_loss) / (max_loss - min_loss)
        self.normalized_losses = normalized.tolist()
        return normalized
    
    def save(self, filename='metrics.json'):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.save_dir, filename)
        
        metrics_dict = {
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'acceptance_rates': self.acceptance_rates,
            'rejection_rates': self.rejection_rates,
            'operational_costs': self.operational_costs,
            'qoe_scores': self.qoe_scores,
            'normalized_rewards': self.normalized_rewards,
            'normalized_losses': self.normalized_losses
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def load(self, filename='metrics.json'):
        """Load metrics from JSON file"""
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"Metrics file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            metrics_dict = json.load(f)
        
        self.episode_rewards = metrics_dict.get('episode_rewards', [])
        self.actor_losses = metrics_dict.get('actor_losses', [])
        self.critic_losses = metrics_dict.get('critic_losses', [])
        self.acceptance_rates = metrics_dict.get('acceptance_rates', [])
        self.rejection_rates = metrics_dict.get('rejection_rates', [])
        self.operational_costs = metrics_dict.get('operational_costs', [])
        self.qoe_scores = metrics_dict.get('qoe_scores', [])
        self.normalized_rewards = metrics_dict.get('normalized_rewards', [])
        self.normalized_losses = metrics_dict.get('normalized_losses', [])
        
        print(f"Metrics loaded from {filepath}")
    
    def print_summary(self):
        """Print summary of metrics"""
        print("\n" + "="*60)
        print("TRAINING METRICS SUMMARY")
        print("="*60)
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Average Reward (last 100): {self.get_recent_avg_reward():.4f}")
        print(f"Average Cost (last 100): {self.get_recent_avg_cost():.4f}")
        print(f"Average Rejection Rate (last 100): {self.get_recent_avg_rejection_rate():.4f}")
        print(f"Average QoE (last 100): {self.get_recent_avg_qoe():.4f}")
        print("="*60 + "\n")