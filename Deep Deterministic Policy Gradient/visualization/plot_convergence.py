"""
Plot convergence curves (normalized reward and training loss)
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


def load_metrics(model_name, scale):
    """Load metrics from JSON file"""
    metrics_path = os.path.join('results', scale, model_name, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics not found at {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def plot_normalized_reward(scale='small', save_dir='results/plots'):
    """
    Plot normalized reward vs episodes for all models
    Corresponds to Figure 2 in the paper
    """
    models = {
        'DDPGA-TS_Proposed': {'color': 'blue', 'label': 'Proposed (DDPGA-TS)'},
        'DDPG-NN': {'color': 'green', 'label': 'DDPG-NN'},
        'DDPG-CNN': {'color': 'red', 'label': 'DDPG-CNN'}
    }
    
    plt.figure(figsize=(10, 6))
    
    for model_name, style in models.items():
        metrics = load_metrics(model_name, scale)
        
        if metrics and len(metrics['normalized_rewards']) > 0:
            rewards = metrics['normalized_rewards']
            episodes = list(range(len(rewards)))
            
            plt.plot(episodes, rewards, 
                    color=style['color'], 
                    label=style['label'], 
                    linewidth=2)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Normalized Reward', fontsize=14)
    plt.title(f'Normalized Reward vs Episodes ({scale.capitalize()} Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'normalized_reward_{scale}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_normalized_loss(scale='small', save_dir='results/plots'):
    """
    Plot normalized training loss vs episodes for all models
    Corresponds to Figure 3 in the paper
    """
    models = {
        'DDPGA-TS_Proposed': {'color': 'blue', 'label': 'Proposed (DDPGA-TS)'},
        'DDPG-NN': {'color': 'green', 'label': 'DDPG-NN'},
        'DDPG-CNN': {'color': 'red', 'label': 'DDPG-CNN'}
    }
    
    plt.figure(figsize=(10, 6))
    
    for model_name, style in models.items():
        metrics = load_metrics(model_name, scale)
        
        if metrics and len(metrics['normalized_losses']) > 0:
            losses = metrics['normalized_losses']
            episodes = list(range(len(losses)))
            
            plt.plot(episodes, losses, 
                    color=style['color'], 
                    label=style['label'], 
                    linewidth=2)
    
    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Normalized Training Loss', fontsize=14)
    plt.title(f'Normalized Training Loss vs Episodes ({scale.capitalize()} Scale)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'normalized_loss_{scale}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_all_convergence(scales=['small', 'medium', 'large'], save_dir='results/plots'):
    """Plot convergence for all scales"""
    for scale in scales:
        print(f"\nPlotting convergence for {scale} scale...")
        plot_normalized_reward(scale, save_dir)
        plot_normalized_loss(scale, save_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot convergence curves')
    parser.add_argument('--scale', type=str, default='all',
                       choices=['all', 'small', 'medium', 'large'],
                       help='Environment scale to plot')
    
    args = parser.parse_args()
    
    if args.scale == 'all':
        plot_all_convergence()
    else:
        plot_normalized_reward(args.scale)
        plot_normalized_loss(args.scale)