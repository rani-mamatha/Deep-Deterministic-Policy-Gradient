"""
Training script for DDPGA-TS and baseline methods
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.edge_cloud_env import EdgeCloudEnvironment
from algorithms.ddpga_ts import DDPGA_TS
from algorithms.ddpg_nn import DDPG_NN
from algorithms.ddpg_cnn import DDPG_CNN
from utils.metrics import MetricsTracker
from config.config import Config
from config.hyperparameters import Hyperparameters


def train_model(algorithm_class, env_config, scale_name, device='cpu', 
                save_dir='saved_models', results_dir='results'):
    """
    Train a model
    
    Args:
        algorithm_class: Algorithm class (DDPGA_TS, DDPG_NN, or DDPG_CNN)
        env_config: Environment configuration
        scale_name: Name of environment scale
        device: Device to use
        save_dir: Directory to save models
        results_dir: Directory to save results
    """
    print(f"\n{'='*80}")
    print(f"Training {algorithm_class.__name__} on {scale_name} environment")
    print(f"{'='*80}\n")
    
    # Create environment
    env = EdgeCloudEnvironment(
        num_edge_nodes=env_config['edge_nodes'],
        num_cloud_nodes=env_config['cloud_nodes'],
        num_jobs=env_config['num_jobs']
    )
    
    # Create algorithm
    algorithm = algorithm_class(env, device=device)
    
    # Create metrics tracker
    model_name = algorithm.name.replace(' ', '_').replace('(', '').replace(')', '')
    metrics_dir = os.path.join(results_dir, scale_name, model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_tracker = MetricsTracker(metrics_dir)
    
    # Training loop
    best_reward = -float('inf')
    num_episodes = Hyperparameters.MAX_EPISODES
    
    for episode in tqdm(range(1, num_episodes + 1), desc=f"Training {model_name}"):
        # Train episode
        episode_reward, actor_loss, critic_loss, env_metrics = algorithm.train_episode()
        
        # Track metrics
        metrics_tracker.add_episode(episode, episode_reward, actor_loss, 
                                    critic_loss, env_metrics)
        
        # Print progress
        if episode % 50 == 0:
            avg_reward = metrics_tracker.get_recent_avg_reward(50)
            avg_cost = metrics_tracker.get_recent_avg_cost(50)
            avg_rejection = metrics_tracker.get_recent_avg_rejection_rate(50)
            avg_qoe = metrics_tracker.get_recent_avg_qoe(50)
            
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Avg Reward (50 eps): {avg_reward:.4f}")
            print(f"  Avg Cost: {avg_cost:.4f}")
            print(f"  Avg Rejection Rate: {avg_rejection:.4f}")
            print(f"  Avg QoE: {avg_qoe:.4f}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            model_save_dir = os.path.join(save_dir, model_name, scale_name)
            os.makedirs(model_save_dir, exist_ok=True)
            model_path = os.path.join(model_save_dir, 'best_model.pth')
            algorithm.save(model_path)
        
        # Save checkpoint periodically
        if episode % Hyperparameters.SAVE_FREQUENCY == 0:
            model_save_dir = os.path.join(save_dir, model_name, scale_name)
            os.makedirs(model_save_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_save_dir, f'checkpoint_ep{episode}.pth')
            algorithm.save(checkpoint_path)
    
    # Normalize metrics for convergence plots
    metrics_tracker.normalize_rewards()
    metrics_tracker.normalize_losses()
    
    # Save metrics
    metrics_tracker.save('metrics.json')
    metrics_tracker.print_summary()
    
    print(f"\n{'='*80}")
    print(f"Training completed for {model_name} on {scale_name}")
    print(f"Best reward: {best_reward:.4f}")
    print(f"{'='*80}\n")
    
    return metrics_tracker


def train_all_models(scale='small', device='cpu'):
    """
    Train all models (DDPGA-TS, DDPG-NN, DDPG-CNN) on specified scale
    
    Args:
        scale: Environment scale ('small', 'medium', 'large')
        device: Device to use ('cpu' or 'cuda')
    """
    # Get environment configuration
    env_config = Config.get_environment_config(scale)
    
    # Train all models
    algorithms = [DDPGA_TS, DDPG_NN, DDPG_CNN]
    
    for algorithm_class in algorithms:
        train_model(
            algorithm_class=algorithm_class,
            env_config=env_config,
            scale_name=scale,
            device=device
        )
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DDPGA-TS and baseline methods')
    parser.add_argument('--scale', type=str, default='small', 
                       choices=['small', 'medium', 'large'],
                       help='Environment scale')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'ddpga_ts', 'ddpg_nn', 'ddpg_cnn'],
                       help='Which model to train')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models(scale=args.scale, device=args.device)
    else:
        env_config = Config.get_environment_config(args.scale)
        
        if args.model == 'ddpga_ts':
            algorithm_class = DDPGA_TS
        elif args.model == 'ddpg_nn':
            algorithm_class = DDPG_NN
        else:
            algorithm_class = DDPG_CNN
        
        train_model(
            algorithm_class=algorithm_class,
            env_config=env_config,
            scale_name=args.scale,
            device=args.device
        )