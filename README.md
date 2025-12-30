# DDPG-MuJoCo

Deep reinforcement learning implementation for MuJoCo environments, specifically trained on Humanoid-v5.

## Project Evolution

This project started with a Deep Deterministic Policy Gradient (DDPG) implementation. However, we encountered stability issues during training - the agent struggled to learn consistently and performance was unreliable.

To address these challenges, we transitioned to Twin Delayed Deep Deterministic Policy Gradient (TD3), which incorporates several improvements over DDPG:
- Twin Q-networks to reduce overestimation bias
- Delayed policy updates for more stable learning
- Target policy smoothing to reduce variance

The TD3 implementation proved much more stable and achieved better training performance.

## Running

Use the provided run scripts:
- `run_ddpg.sh` - Run DDPG training
- `run_td3.sh` - Run TD3 training (recommended)
