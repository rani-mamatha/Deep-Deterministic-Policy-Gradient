"""
Compare performance of all models across different scales
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_metrics(model_name, scale):
    """Load metrics from JSON file"""
    metrics_path = os.path.join('results', scale, model_name, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        print(f"Warning: Metrics not found at {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def compare_convergence(scales=['small', 'medium', 'large']):
    """
    Compare convergence across models and scales
    """
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    
    print("\n" + "="*80)
    print("CONVERGENCE COMPARISON")
    print("="*80 + "\n")
    
    for scale in scales:
        print(f"\n{scale.upper()} Scale Environment:")
        print("-" * 60)
        
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics and len(metrics['normalized_rewards']) > 0:
                # Get final normalized reward (convergence)
                final_reward = metrics['normalized_rewards'][-1]
                final_loss = metrics['normalized_losses'][-1] if metrics['normalized_losses'] else 0
                
                print(f"{model:25s}: Final Reward={final_reward:.4f}, Final Loss={final_loss:.4f}")


def compare_performance(scales=['small', 'medium', 'large']):
    """
    Compare performance metrics across models and scales
    """
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80 + "\n")
    
    # Prepare data for table
    data = []
    
    for scale in scales:
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics:
                # Calculate average metrics from last 100 episodes
                last_100_costs = metrics['operational_costs'][-100:] if len(metrics['operational_costs']) >= 100 else metrics['operational_costs']
                last_100_rejections = metrics['rejection_rates'][-100:] if len(metrics['rejection_rates']) >= 100 else metrics['rejection_rates']
                last_100_qoe = metrics['qoe_scores'][-100:] if len(metrics['qoe_scores']) >= 100 else metrics['qoe_scores']
                
                avg_cost = np.mean(last_100_costs) if last_100_costs else 0
                avg_rejection = np.mean(last_100_rejections) if last_100_rejections else 0
                avg_qoe = np.mean(last_100_qoe) if last_100_qoe else 0
                
                data.append({
                    'Scale': scale,
                    'Model': model,
                    'Avg Cost': avg_cost,
                    'Avg Rejection Rate': avg_rejection,
                    'Avg QoE': avg_qoe
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display tables for each metric
    print("\n--- AVERAGE OPERATIONAL COST ---")
    cost_table = df.pivot(index='Model', columns='Scale', values='Avg Cost')
    print(cost_table.to_string())
    
    print("\n--- AVERAGE REJECTION RATE ---")
    rejection_table = df.pivot(index='Model', columns='Scale', values='Avg Rejection Rate')
    print(rejection_table.to_string())
    
    print("\n--- AVERAGE QUALITY OF EXPERIENCE (QoE) ---")
    qoe_table = df.pivot(index='Model', columns='Scale', values='Avg QoE')
    print(qoe_table.to_string())
    
    # Save to CSV
    results_dir = 'results/comparison'
    os.makedirs(results_dir, exist_ok=True)
    
    cost_table.to_csv(os.path.join(results_dir, 'operational_cost.csv'))
    rejection_table.to_csv(os.path.join(results_dir, 'rejection_rate.csv'))
    qoe_table.to_csv(os.path.join(results_dir, 'qoe.csv'))
    
    print(f"\nComparison tables saved to {results_dir}/")


def generate_summary_report():
    """Generate comprehensive summary report"""
    scales = ['small', 'medium', 'large']
    models = ['DDPGA-TS_Proposed', 'DDPG-NN', 'DDPG-CNN']
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80 + "\n")
    
    compare_convergence(scales)
    compare_performance(scales)
    
    # Highlight best model for each metric
    print("\n" + "="*80)
    print("BEST PERFORMING MODELS")
    print("="*80 + "\n")
    
    for scale in scales:
        print(f"\n{scale.upper()} Scale:")
        print("-" * 60)
        
        costs = {}
        rejections = {}
        qoes = {}
        
        for model in models:
            metrics = load_metrics(model, scale)
            
            if metrics:
                last_100_costs = metrics['operational_costs'][-100:]
                last_100_rejections = metrics['rejection_rates'][-100:]
                last_100_qoe = metrics['qoe_scores'][-100:]
                
                costs[model] = np.mean(last_100_costs) if last_100_costs else float('inf')
                rejections[model] = np.mean(last_100_rejections) if last_100_rejections else float('inf')
                qoes[model] = np.mean(last_100_qoe) if last_100_qoe else 0
        
        best_cost = min(costs, key=costs.get)
        best_rejection = min(rejections, key=rejections.get)
        best_qoe = max(qoes, key=qoes.get)
        
        print(f"  Lowest Cost: {best_cost} ({costs[best_cost]:.4f})")
        print(f"  Lowest Rejection Rate: {best_rejection} ({rejections[best_rejection]:.4f})")
        print(f"  Highest QoE: {best_qoe} ({qoes[best_qoe]:.4f})")


if __name__ == "__main__":
    generate_summary_report()