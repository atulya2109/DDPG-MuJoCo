import gymnasium as gym
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime

from lib.agent import DDPGAgent


def train_ddpg(
    env_name="Humanoid-v5",
    num_episodes=2000,
    max_steps=1000,
    batch_size=256,
    warmup_steps=10000,
    device="mps",
    save_interval=1000,
    log_dir="runs",
    checkpoint_dir="checkpoints",
):
    # Create environment
    env = gym.make(env_name)
    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")

    # Create agent
    agent = DDPGAgent(
        obs_size=obs_dim,
        action_size=action_dim,
        device=device,
        gamma=0.99,
        tau=0.0001,
        actor_lr=1e-5,  # Slowed down to prevent critic from lagging
        critic_lr=1e-4,  # Further reduced for stability
        buffer_capacity=1000000,
        policy_delay=3,  # TD3: Update actor every 3 critic updates
        target_noise=0.2,  # TD3: Target policy smoothing
        noise_clip=0.5,  # TD3: Noise clipping
    )

    # Setup TensorBoard and checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f"{log_dir}/{env_name}_{timestamp}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    episode_rewards = []
    total_steps = 0
    start_time = time.time()

    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Algorithm: TD3 (Twin Delayed DDPG)")
    print(f"Warmup steps: {warmup_steps} (random exploration)")
    print(f"TensorBoard logs: {log_dir}/{env_name}_{timestamp}")
    print(f"Checkpoints: {checkpoint_dir}/")
    print(f"Buffer capacity: {agent.buffer.capacity}")
    print(f"Initial buffer size: {len(agent.buffer)}")
    print(f"Reward scaling: 0.1x")
    print(f"Critic LR: {agent.critic_1_optimizer.param_groups[0]['lr']:.2e}")
    print(f"Actor LR: {agent.actor_optimizer.param_groups[0]['lr']:.2e}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    pbar = tqdm(range(num_episodes), desc="Training", unit="episode")

    for episode in pbar:
        state, _ = env.reset()
        agent.reset_noise()
        episode_reward = 0.0
        episode_steps = 0
        episode_actor_loss = 0.0
        episode_critic_loss = 0.0
        training_steps = 0
        actor_updates = 0

        for _ in range(max_steps):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Scale reward to prevent large gradients
            reward_scaled = float(reward) * 0.1
            agent.store_transition(
                state, action, reward_scaled, next_state, float(done)
            )

            if total_steps >= warmup_steps:
                actor_loss, critic_loss = agent.train(batch_size)
                if critic_loss is not None:
                    episode_critic_loss += critic_loss
                    training_steps += 1
                if actor_loss is not None:
                    episode_actor_loss += actor_loss
                    actor_updates += 1

            episode_reward += float(reward)
            episode_steps += 1
            total_steps += 1
            state = next_state

            if done:
                break

        mean_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
        episode_rewards.append(mean_reward)
        avg_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes

        # Calculate elapsed time
        elapsed_hours = (time.time() - start_time) / 3600

        writer.add_scalar("Mean_Reward/Hours", mean_reward, elapsed_hours)
        writer.add_scalar("Episode/Avg_Mean_Reward_100", avg_reward, episode)
        writer.add_scalar("Training/Time_Hours", elapsed_hours, episode)

        if training_steps > 0:
            avg_critic_loss = episode_critic_loss / training_steps
            writer.add_scalar("Loss/Critic", avg_critic_loss, episode)

        if actor_updates > 0:
            avg_actor_loss = episode_actor_loss / actor_updates
            writer.add_scalar("Loss/Actor", avg_actor_loss, episode)

        # Update progress bar
        pbar.set_postfix(
            {
                "mean_r": f"{mean_reward:.2f}",
                "avg_100": f"{avg_reward:.2f}",
                "hrs": f"{elapsed_hours:.2f}",
                "buffer": f"{len(agent.buffer)//1000}k",
            }
        )

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            save_checkpoint(agent, episode + 1, episode_rewards, checkpoint_dir)
            tqdm.write(f"  → Checkpoint saved at episode {episode + 1}")

    env.close()
    writer.close()

    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Total time: {hours}h {minutes}m")
    print(f"Final avg reward (100 ep): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best avg reward (100 ep): {max([np.mean(episode_rewards[max(0,i-99):i+1]) for i in range(len(episode_rewards))]):.2f}")
    print(f"View results: tensorboard --logdir={log_dir}")
    print("="*60)

    return agent, episode_rewards


def save_checkpoint(agent, episode, episode_rewards, checkpoint_dir="checkpoints"):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        "episode": episode,
        "actor_state_dict": agent.actor.state_dict(),
        "critic_1_state_dict": agent.critic_1.state_dict(),
        "critic_2_state_dict": agent.critic_2.state_dict(),
        "target_actor_state_dict": agent.target_actor.state_dict(),
        "target_critic_1_state_dict": agent.target_critic_1.state_dict(),
        "target_critic_2_state_dict": agent.target_critic_2.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_1_optimizer_state_dict": agent.critic_1_optimizer.state_dict(),
        "critic_2_optimizer_state_dict": agent.critic_2_optimizer.state_dict(),
        "episode_rewards": episode_rewards,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}.pt")
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(agent, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic_1.load_state_dict(checkpoint["critic_1_state_dict"])
    agent.critic_2.load_state_dict(checkpoint["critic_2_state_dict"])
    agent.target_actor.load_state_dict(checkpoint["target_actor_state_dict"])
    agent.target_critic_1.load_state_dict(checkpoint["target_critic_1_state_dict"])
    agent.target_critic_2.load_state_dict(checkpoint["target_critic_2_state_dict"])
    agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    agent.critic_1_optimizer.load_state_dict(
        checkpoint["critic_1_optimizer_state_dict"]
    )
    agent.critic_2_optimizer.load_state_dict(
        checkpoint["critic_2_optimizer_state_dict"]
    )
    return checkpoint["episode"], checkpoint["episode_rewards"]


if __name__ == "__main__":
    # Run training
    agent, rewards = train_ddpg(
        env_name="Humanoid-v5",
        num_episodes=100000,
        max_steps=1000,
        batch_size=512,  # Increased for more stable gradients
        warmup_steps=10000,
        device="mps",
        save_interval=1000,
    )

    # Save final model
    save_checkpoint(agent, "final", rewards, checkpoint_dir="checkpoints")
    print("Final model saved to checkpoints/checkpoint_epfinal.pt")
