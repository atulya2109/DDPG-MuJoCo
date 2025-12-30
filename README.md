# DDPG-MuJoCo

Deep reinforcement learning implementation for MuJoCo environments, specifically trained on Humanoid-v5.

## Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/92685f0c-cdde-4b3c-bb98-1c6a2cf01167" width="400" />
</div>


## Project Evolution

This project started with a Deep Deterministic Policy Gradient (DDPG) implementation. However, stability issues were encountered during training - the agent struggled to learn consistently and performance was unreliable.

To address these challenges, the implementation was transitioned to Twin Delayed Deep Deterministic Policy Gradient (TD3), which incorporates several improvements over DDPG:
- Twin Q-networks to reduce overestimation bias
- Delayed policy updates for more stable learning
- Target policy smoothing to reduce variance

The TD3 implementation proved much more stable and achieved better training performance.

## Running

### Training

```bash
python train.py
```

The training script uses hardcoded parameters. To modify training settings, edit the parameters in train.py:210-220:
- `env_name`: Environment to train on (default: Humanoid-v5)
- `num_episodes`: Number of training episodes (default: 100000)
- `max_steps`: Max steps per episode (default: 1000)
- `batch_size`: Batch size for training (default: 512)
- `warmup_steps`: Random exploration steps before training (default: 10000)
- `device`: Device to use (default: mps)
- `save_interval`: Episodes between checkpoint saves (default: 1000)

### Visualization

```bash
python visualize.py [OPTIONS]
```

Options:
- `--checkpoint PATH`: Path to checkpoint file (default: checkpoints/checkpoint_epfinal.pt)
- `--env ENV`: Environment name (default: Humanoid-v5)
- `--episodes N`: Number of episodes to run (default: 5)
- `--device DEVICE`: Device to use - mps/cuda/cpu (default: mps)
- `--save-video`: Save video of one episode to videos/ folder
- `--fps N`: Video FPS (default: 30)
- `--compare PATH1 PATH2 ...`: Compare multiple checkpoints

Examples:
```bash
# Visualize with default checkpoint
python visualize.py

# Save a video
python visualize.py --save-video

# Visualize specific checkpoint
python visualize.py --checkpoint checkpoints/checkpoint_ep2000.pt --episodes 10

# Compare multiple checkpoints
python visualize.py --compare checkpoints/checkpoint_ep1000.pt checkpoints/checkpoint_ep2000.pt
```
