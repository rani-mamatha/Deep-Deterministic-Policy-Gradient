"""
Plot performance comparison (cost, rejection rate, QoE)
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


def plot_operational_cost(save_dir='results/plots'):
    """
    Plot average operational cost across all scales
    Corresponds to Figure 4 in the paper
    """
    scales = ['small', 'medium', 'large']
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    model_labels = ['Proposed (DDPGA-TS)', 'DDPG-NN', 'DDPG-CNN']
    
    # Collect data
    data = {model: [] for model in models}
    
    for scale in scales:
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics:
                last_100 = metrics['operational_costs'][-100:]
                avg_cost = np.mean(last_100) if last_100 else 0
                data[model].append(avg_cost)
            else:
                data[model].append(0)
    
    # Plot
    x = np.arange(len(scales))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, (model, label) in enumerate(zip(models, model_labels)):
        ax.bar(x + i*width, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Environment Scale', fontsize=14)
    ax.set_ylabel('Average Operational Cost', fontsize=14)
    ax.set_title('Average Operational Cost Comparison', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in scales])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'operational_cost_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_rejection_rate(save_dir='results/plots'):
    """
    Plot average rejection rate across all scales
    Corresponds to Figure 5 in the paper
    """
    scales = ['small', 'medium', 'large']
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    model_labels = ['Proposed (DDPGA-TS)', 'DDPG-NN', 'DDPG-CNN']
    
    # Collect data
    data = {model: [] for model in models}
    
    for scale in scales:
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics:
                last_100 = metrics['rejection_rates'][-100:]
                avg_rejection = np.mean(last_100) * 100 if last_100 else 0  # Convert to percentage
                data[model].append(avg_rejection)
            else:
                data[model].append(0)
    
    # Plot
    x = np.arange(len(scales))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, (model, label) in enumerate(zip(models, model_labels)):
        ax.bar(x + i*width, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Environment Scale', fontsize=14)
    ax.set_ylabel('Average Rejection Rate (%)', fontsize=14)
    ax.set_title('Average Rejection Rate Comparison', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in scales])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rejection_rate_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_qoe(save_dir='results/plots'):
    """
    Plot Quality of Experience across all scales
    Corresponds to Figure 6 in the paper
    """
    scales = ['small', 'medium', 'large']
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    model_labels = ['Proposed (DDPGA-TS)', 'DDPG-NN', 'DDPG-CNN']
    
    # Collect data
    data = {model: [] for model in models}
    
    for scale in scales:
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics:
                last_100 = metrics['qoe_scores'][-100:]
                avg_qoe = np.mean(last_100) if last_100 else 0
                data[model].append(avg_qoe)
            else:
                data[model].append(0)
    
    # Plot
    x = np.arange(len(scales))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, (model, label) in enumerate(zip(models, model_labels)):
        ax.bar(x + i*width, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Environment Scale', fontsize=14)
    ax.set_ylabel('Quality of Experience (QoE)', fontsize=14)
    ax.set_title('Quality of Experience (QoE) Comparison', fontsize=16)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in scales])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'qoe_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_all_performance(save_dir='results/plots'):
    """Plot all performance metrics"""
    print("\nGenerating performance comparison plots...")
    plot_operational_cost(save_dir)
    plot_rejection_rate(save_dir)
    plot_qoe(save_dir)
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    plot_all_performance()