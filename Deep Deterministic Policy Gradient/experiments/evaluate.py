"""
Evaluation script for trained models
"""

import os
import sys
import torch
import numpy as np
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.edge_cloud_env import EdgeCloudEnvironment
from algorithms.ddpga_ts import DDPGA_TS
from algorithms.ddpg_nn import DDPG_NN
from algorithms.ddpg_cnn import DDPG_CNN
from config.config import Config


def evaluate_model(algorithm_class, env_config, scale_name, model_path, 
                   device='cpu', num_eval_episodes=10):
    """
    Evaluate a trained model
    
    Args:
        algorithm_class: Algorithm class
        env_config: Environment configuration
        scale_name: Name of environment scale
        model_path: Path to saved model
        device: Device to use
        num_eval_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {algorithm_class.__name__} on {scale_name} environment")
    
    # Create environment
    env = EdgeCloudEnvironment(
        num_edge_nodes=env_config['edge_nodes'],
        num_cloud_nodes=env_config['cloud_nodes'],
        num_jobs=env_config['num_jobs']
    )
    
    # Create algorithm
    algorithm = algorithm_class(env, device=device)
    
    # Load model
    if os.path.exists(model_path):
        algorithm.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
        return None
    
    # Evaluation
    eval_rewards = []
    eval_costs = []
    eval_rejection_rates = []
    eval_qoe = []
    
    for ep in range(num_eval_episodes):
        episode_reward, env_metrics = algorithm.evaluate()
        
        eval_rewards.append(episode_reward)
        eval_costs.append(env_metrics['avg_cost'])
        eval_rejection_rates.append(env_metrics['rejection_rate'])
        eval_qoe.append(env_metrics['qoe'])
        
        print(f"  Episode {ep+1}/{num_eval_episodes}: "
              f"Reward={episode_reward:.2f}, Cost={env_metrics['avg_cost']:.4f}, "
              f"Rejection={env_metrics['rejection_rate']:.4f}, QoE={env_metrics['qoe']:.4f}")
    
    # Compute statistics
    results = {
        'model': algorithm.name,
        'scale': scale_name,
        'num_episodes': num_eval_episodes,
        'avg_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'avg_cost': float(np.mean(eval_costs)),
        'std_cost': float(np.std(eval_costs)),
        'avg_rejection_rate': float(np.mean(eval_rejection_rates)),
        'std_rejection_rate': float(np.std(eval_rejection_rates)),
        'avg_qoe': float(np.mean(eval_qoe)),
        'std_qoe': float(np.std(eval_qoe))
    }
    
    print(f"\nEvaluation Results for {algorithm.name}:")
    print(f"  Average Reward: {results['avg_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"  Average Cost: {results['avg_cost']:.4f} ± {results['std_cost']:.4f}")
    print(f"  Average Rejection Rate: {results['avg_rejection_rate']:.4f} ± {results['std_rejection_rate']:.4f}")
    print(f"  Average QoE: {results['avg_qoe']:.4f} ± {results['std_qoe']:.4f}")
    
    return results


def evaluate_all_models(scale='small', device='cpu', num_eval_episodes=10):
    """
    Evaluate all trained models
    
    Args:
        scale: Environment scale
        device: Device to use
        num_eval_episodes: Number of evaluation episodes
    """
    env_config = Config.get_environment_config(scale)
    
    algorithms = [
        (DDPGA_TS, 'DDPGA-TS_Proposed'),
        (DDPG_NN, 'DDPG-NN'),
        (DDPG_CNN, 'DDPG-CNN')
    ]
    
    all_results = []
    
    for algorithm_class, model_name in algorithms:
        model_path = os.path.join('saved_models', model_name, scale, 'best_model.pth')
        
        results = evaluate_model(
            algorithm_class=algorithm_class,
            env_config=env_config,
            scale_name=scale,
            model_path=model_path,
            device=device,
            num_eval_episodes=num_eval_episodes
        )
        
        if results:
            all_results.append(results)
    
    # Save combined results
    results_dir = os.path.join('results', scale)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation results saved to {results_dir}/evaluation_results.json")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--scale', type=str, default='small',
                       choices=['small', 'medium', 'large'],
                       help='Environment scale')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    evaluate_all_models(
        scale=args.scale,
        device=args.device,
        num_eval_episodes=args.episodes
    )