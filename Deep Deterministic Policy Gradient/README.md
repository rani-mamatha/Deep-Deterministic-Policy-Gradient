# DDPGA-TS: Deep Deterministic Policy Gradient Algorithm for Dynamic Task Scheduling

Implementation of the paper: "Deep Deterministic Policy Gradient Algorithm for Dynamic Task Scheduling in Edge-Cloud Environment Using Reinforcement Learning"

## 📋 Overview

This project implements **DDPGA-TS**, a novel reinforcement learning algorithm for dynamic task scheduling in edge-cloud computing environments. The algorithm uses:

- **Conv1D** for local feature extraction
- **GRU** for temporal dependencies learning
- **Attention mechanism** for prediction enhancement
- **Novel pruning strategy** for action space reduction

## 🏗️ Project Structure
```
objective 2/
│
├── config/                          # Configuration files
│   ├── __init__.py
│   ├── config.py                    # Environment configurations
│   └── hyperparameters.py           # Training hyperparameters
│
├── models/                          # Neural network models
│   ├── __init__.py
│   ├── actor_network.py             # Actor network (policy)
│   ├── critic_network.py            # Critic network (Q-function)
│   └── ddpg_agent.py                # DDPG agent implementation
│
├── environment/                     # Edge-Cloud environment
│   ├── __init__.py
│   ├── job.py                       # Job class
│   ├── resource_manager.py          # Resource management
│   └── edge_cloud_env.py            # Environment implementation
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   ├── replay_buffer.py             # Experience replay buffer
│   ├── noise.py                     # Ornstein-Uhlenbeck noise
│   ├── metrics.py                   # Performance metrics tracker
│   └── pruning.py                   # Action space pruning (Novel!)
│
├── algorithms/                      # Algorithm implementations
│   ├── __init__.py
│   ├── ddpga_ts.py                  # Proposed DDPGA-TS algorithm
│   ├── ddpg_nn.py                   # DDPG-NN baseline
│   └── ddpg_cnn.py                  # DDPG-CNN baseline
│
├── experiments/                     # Training and evaluation scripts
│   ├── __init__.py
│   ├── train.py                     # Training script
│   ├── evaluate.py                  # Evaluation script
│   └── compare_models.py            # Model comparison
│
├── visualization/                   # Plotting and visualization
│   ├── __init__.py
│   ├── plot_convergence.py          # Convergence plots
│   └── plot_performance.py          # Performance comparison plots
│
├── results/                         # Results directory (auto-created)
│   ├── small/                       # Small scale results
│   ├── medium/                      # Medium scale results
│   ├── large/                       # Large scale results
│   └── plots/                       # Generated plots
│
├── saved_models/                    # Trained models (auto-created)
│   ├── DDPGA-TS_Proposed/
│   ├── DDPG-NN/
│   └── DDPG-CNN/
│
├── logs/                            # Training logs (auto-created)
│
├── requirements.txt                 # Python dependencies
├── main.py                          # Main execution script
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or navigate to the project directory
cd "C:\Users\krake\Downloads\mamatha rani\objective 2"

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Experiment (Small Scale)
```bash
python main.py --mode full --scale small
```

This will:
1. Setup directories
2. Train all three models (DDPGA-TS, DDPG-NN, DDPG-CNN)
3. Evaluate all models
4. Generate comparison report
5. Create all visualizations

### 3. Run All Scales
```bash
python main.py --mode full --scale all
```

## 📊 Environment Scales

| Scale  | Edge Nodes | Cloud Nodes | Jobs      |
|--------|-----------|-------------|-----------|
| Small  | 10        | 10          | 10,000    |
| Medium | 30        | 30          | 100,000   |
| Large  | 50        | 50          | 1,000,000 |

## 🎯 Usage Examples

### Train Only DDPGA-TS on Medium Scale
```bash
python main.py --mode train --model ddpga_ts --scale medium
```

### Train All Models on Large Scale
```bash
python main.py --mode train --scale large
```

### Evaluate Models
```bash
python main.py --mode evaluate --scale small --eval-episodes 20
```

### Generate Visualizations
```bash
python main.py --mode visualize --scale all
```

### Compare Models
```bash
python main.py --mode compare
```

### Use GPU (if available)
```bash
python main.py --mode full --scale small --device cuda
```

## 📈 Performance Metrics

The implementation tracks and compares the following metrics:

1. **Convergence Metrics**
   - Normalized Reward
   - Normalized Training Loss

2. **Performance Metrics**
   - Average Operational Cost
   - Average Rejection Rate
   - Quality of Experience (QoE)

## 🔬 Key Features

### Novel Contributions (DDPGA-TS)

1. **Action Space Pruning Strategy**
   - Continuously monitors resource utilization
   - Reduces action space by pruning low-utility resources
   - Improves convergence and performance
   - Enhances load balancing

2. **Advanced Neural Architecture**
   - Conv1D for local feature extraction
   - GRU for temporal dependencies
   - Multi-head attention mechanism
   - Better than standard NN and CNN approaches

### Baseline Methods

1. **DDPG-NN**: Standard DDPG with fully connected networks
2. **DDPG-CNN**: DDPG with Conv1D only (no GRU or Attention)

## 📊 Expected Results

Based on the paper, DDPGA-TS should outperform baselines:

### Small Scale Environment
- **Operational Cost**: DDPGA-TS (1.4) < DDPG-NN (1.9) < DDPG-CNN (2.1)
- **Rejection Rate**: DDPGA-TS (4) < DDPG-NN (15) < DDPG-CNN (16)
- **QoE**: DDPGA-TS (0.62) > DDPG-NN (0.54) > DDPG-CNN (0.48)

### Medium Scale Environment
- **Operational Cost**: DDPGA-TS (1.3) < DDPG-NN (2.1) < DDPG-CNN (2.3)
- **Rejection Rate**: DDPGA-TS (3) < DDPG-NN (17) < DDPG-CNN (20)
- **QoE**: DDPGA-TS (0.69) > DDPG-NN (0.48) > DDPG-CNN (0.40)

### Large Scale Environment
- **Operational Cost**: DDPGA-TS (1.1) < DDPG-NN (2.3) < DDPG-CNN (2.4)
- **Rejection Rate**: DDPGA-TS (2) < DDPG-NN (23) < DDPG-CNN (24)
- **QoE**: DDPGA-TS (0.70) > DDPG-NN (0.40) > DDPG-CNN (0.28)

## 🔧 Configuration

### Hyperparameters (in `config/hyperparameters.py`)
```python
ACTOR_LR = 0.0001           # Actor learning rate
CRITIC_LR = 0.001           # Critic learning rate
GAMMA = 0.99                # Discount factor
TAU = 0.001                 # Soft update parameter
BATCH_SIZE = 128            # Training batch size
BUFFER_SIZE = 100000        # Replay buffer size
MAX_EPISODES = 1000         # Training episodes
```

### Environment Parameters (in `config/config.py`)
```python
BANDWIDTH_EDGE_CLOUD = 1e9  # 1 Gbps
PROPAGATION_TIME_EDGE = 0.005  # 5ms
PROPAGATION_TIME_CLOUD = 0.050  # 50ms
```

## 📁 Output Files

After running experiments, you'll find:

### Results
```
results/
├── small/
│   ├── DDPGA-TS_Proposed/metrics.json
│   ├── DDPG-NN/metrics.json
│   └── DDPG-CNN/metrics.json
├── plots/
│   ├── normalized_reward_small.png
│   ├── normalized_loss_small.png
│   ├── operational_cost_comparison.png
│   ├── rejection_rate_comparison.png
│   └── qoe_comparison.png
└── comparison/
    ├── operational_cost.csv
    ├── rejection_rate.csv
    └── qoe.csv
```

### Saved Models
```
saved_models/
├── DDPGA-TS_Proposed/
│   └── small/best_model.pth
├── DDPG-NN/
│   └── small/best_model.pth
└── DDPG-CNN/
    └── small/best_model.pth
```

## 🐛 Troubleshooting

### Memory Issues
If you encounter memory issues with large scale:
```bash
# Reduce batch size in config/hyperparameters.py
BATCH_SIZE = 64  # Instead of 128
```

### Slow Training
```bash
# Use GPU if available
python main.py --mode train --scale small --device cuda

# Or reduce number of jobs for testing
# Edit config/config.py and reduce num_jobs
```

### Missing Dependencies
```bash
pip install --upgrade -r requirements.txt
```

## 📚 Algorithm Details

### Reward Function
```
R(j) = Gain(j) - Cost(j)
```

### Gain Function
```
Gain(j) = δ × (deadline - RTT)  if RTT ≤ deadline
```

### Cost Function
```
Cost(j) = (bandwidth_cost × bandwidth × RTT) + (VM_cost × CPU × RTT)
```

### RTT (Round Trip Time)
- **Edge**: 2 × (5ms + data/bandwidth + processing_time)
- **Cloud**: 2 × (50ms + data/bandwidth + processing_time)

## 🤝 Contributing

This is a research implementation. Feel free to:
- Report issues
- Suggest improvements
- Extend the implementation

## 📄 Citation

If you use this code, please cite:
```
Deep Deterministic Policy Gradient Algorithm for Dynamic Task Scheduling 
in Edge-Cloud Environment Using Reinforcement Learning
```

## 📧 Contact

For questions or issues, please open an issue in the repository.

## 🎓 Acknowledgments

This implementation is based on the research paper on DDPGA-TS for edge-cloud task scheduling.

## 📝 License

This project is for academic and research purposes.

---

**Happy Experimenting! 🚀**