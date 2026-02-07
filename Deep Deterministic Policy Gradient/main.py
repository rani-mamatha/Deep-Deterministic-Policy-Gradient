"""
Main script to run the complete DDPGA-TS experiment pipeline
"""

import os
import sys
import argparse
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.train import train_all_models, train_model
from experiments.evaluate import evaluate_all_models
from experiments.compare_models import generate_summary_report
from visualization.plot_convergence import plot_all_convergence
from visualization.plot_performance import plot_all_performance
from config.config import Config
from algorithms.ddpga_ts import DDPGA_TS
from algorithms.ddpg_nn import DDPG_NN
from algorithms.ddpg_cnn import DDPG_CNN


def setup_directories():
    """Create necessary directories"""
    directories = [
        'results/small',
        'results/medium',
        'results/large',
        'results/plots',
        'results/comparison',
        'saved_models/DDPGA-TS_Proposed/small',
        'saved_models/DDPGA-TS_Proposed/medium',
        'saved_models/DDPGA-TS_Proposed/large',
        'saved_models/DDPG-NN/small',
        'saved_models/DDPG-NN/medium',
        'saved_models/DDPG-NN/large',
        'saved_models/DDPG-CNN/small',
        'saved_models/DDPG-CNN/medium',
        'saved_models/DDPG-CNN/large',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully!")


def run_full_experiment(scale='small', device='cpu'):
    """
    Run the complete experiment pipeline:
    1. Train all models
    2. Evaluate all models
    3. Generate comparison report
    4. Create all visualizations
    """
    print("\n" + "="*80)
    print(f"RUNNING FULL EXPERIMENT PIPELINE FOR {scale.upper()} SCALE")
    print("="*80 + "\n")
    
    # Step 1: Setup directories
    print("\n[Step 1/5] Setting up directories...")
    setup_directories()
    
    # Step 2: Train all models
    print("\n[Step 2/5] Training all models...")
    train_all_models(scale=scale, device=device)
    
    # Step 3: Evaluate all models
    print("\n[Step 3/5] Evaluating all models...")
    evaluate_all_models(scale=scale, device=device, num_eval_episodes=10)
    
    # Step 4: Generate comparison report
    print("\n[Step 4/5] Generating comparison report...")
    generate_summary_report()
    
    # Step 5: Create visualizations
    print("\n[Step 5/5] Creating visualizations...")
    plot_all_convergence(scales=[scale])
    plot_all_performance()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    print(f"Results saved in: results/{scale}/")
    print(f"Models saved in: saved_models/")
    print(f"Plots saved in: results/plots/")


def run_all_scales(device='cpu'):
    """Run experiments for all environment scales"""
    scales = ['small', 'medium', 'large']
    
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS FOR ALL SCALES")
    print("="*80 + "\n")
    
    for scale in scales:
        run_full_experiment(scale=scale, device=device)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='DDPGA-TS: Deep Deterministic Policy Gradient Algorithm for Dynamic Task Scheduling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment on small scale
  python main.py --mode full --scale small
  
  # Train only DDPGA-TS on medium scale
  python main.py --mode train --model ddpga_ts --scale medium
  
  # Evaluate all models on large scale
  python main.py --mode evaluate --scale large
  
  # Generate visualizations
  python main.py --mode visualize
  
  # Run all scales
  python main.py --mode full --scale all
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'train', 'evaluate', 'visualize', 'compare'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--scale',
        type=str,
        default='small',
        choices=['small', 'medium', 'large', 'all'],
        help='Environment scale'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['all', 'ddpga_ts', 'ddpg_nn', 'ddpg_cnn'],
        help='Which model to train (only used in train mode)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print("\n" + "="*80)
    print("DDPGA-TS: Deep Deterministic Policy Gradient Algorithm")
    print("for Dynamic Task Scheduling in Edge-Cloud Environment")
    print("="*80 + "\n")
    
    # Execute based on mode
    if args.mode == 'full':
        if args.scale == 'all':
            run_all_scales(device=args.device)
        else:
            run_full_experiment(scale=args.scale, device=args.device)
    
    elif args.mode == 'train':
        setup_directories()
        if args.scale == 'all':
            for scale in ['small', 'medium', 'large']:
                if args.model == 'all':
                    train_all_models(scale=scale, device=args.device)
                else:
                    env_config = Config.get_environment_config(scale)
                    algorithm_class = {
                        'ddpga_ts': DDPGA_TS,
                        'ddpg_nn': DDPG_NN,
                        'ddpg_cnn': DDPG_CNN
                    }[args.model]
                    train_model(algorithm_class, env_config, scale, args.device)
        else:
            if args.model == 'all':
                train_all_models(scale=args.scale, device=args.device)
            else:
                env_config = Config.get_environment_config(args.scale)
                algorithm_class = {
                    'ddpga_ts': DDPGA_TS,
                    'ddpg_nn': DDPG_NN,
                    'ddpg_cnn': DDPG_CNN
                }[args.model]
                train_model(algorithm_class, env_config, args.scale, args.device)
    
    elif args.mode == 'evaluate':
        if args.scale == 'all':
            for scale in ['small', 'medium', 'large']:
                evaluate_all_models(scale=scale, device=args.device, 
                                  num_eval_episodes=args.eval_episodes)
        else:
            evaluate_all_models(scale=args.scale, device=args.device,
                              num_eval_episodes=args.eval_episodes)
    
    elif args.mode == 'visualize':
        if args.scale == 'all':
            plot_all_convergence(scales=['small', 'medium', 'large'])
        else:
            plot_all_convergence(scales=[args.scale])
        plot_all_performance()
    
    elif args.mode == 'compare':
        generate_summary_report()
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()