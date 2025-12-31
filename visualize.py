import gymnasium as gym
import torch
import numpy as np
import argparse
from lib.agent import DDPGAgent
from wrapppers import HumanoidPDWrapper


def visualize_agent(
    checkpoint_path,
    env_name="Humanoid-v5",
    num_episodes=5,
    max_steps=1000,
    device="mps",
    render_mode="human",
    save_video=False,
    video_fps=30,
):
    """Visualize trained DDPG agent"""

    # Create environment with rendering
    if save_video:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        env = gym.make(env_name, render_mode="rgb_array", width=1920, height=1080)
        env = HumanoidPDWrapper(env, kp=0.5, kd=0.05)
        # Wrap with video recorder
        env = gym.wrappers.RecordVideo(
            env,
            video_folder="videos",
            episode_trigger=lambda episode_id: episode_id == 0,  # Record only first episode
            name_prefix=f"ddpg_{env_name}_{timestamp}",
            fps=video_fps,
        )
        # Override num_episodes to 1 when saving video
        num_episodes = 1
        print(f"Recording video to: videos/ddpg_{env_name}_{timestamp}-episode-0.mp4")
    else:
        env = gym.make(env_name, render_mode=render_mode, width=1920, height=1080)
        env = HumanoidPDWrapper(env, kp=0.5, kd=0.05)

    assert env.observation_space.shape is not None
    assert env.action_space.shape is not None
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print("-" * 60)

    # Create agent
    agent = DDPGAgent(
        obs_size=obs_dim,
        action_size=action_dim,
        device=device,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.actor.eval()  # Set to evaluation mode

    print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}")
    print(f"Running {num_episodes} episodes...\n")

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_steps):
            # Select action (NO NOISE for evaluation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.actor(state_tensor).cpu().numpy().squeeze()

            # Clip to action bounds
            action = np.clip(action, -1.0, 1.0)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_steps += 1
            state = next_state

            if done:
                break

        mean_reward = episode_reward / episode_steps if episode_steps > 0 else 0.0
        episode_rewards.append(mean_reward)
        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Steps: {episode_steps:4d} | "
            f"Mean Reward: {mean_reward:6.2f}"
        )

    env.close()

    # Summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Mean Reward per Step)")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Average mean reward: {np.mean(episode_rewards):.2f}")
    print(f"Std mean reward: {np.std(episode_rewards):.2f}")
    print(f"Min mean reward: {np.min(episode_rewards):.2f}")
    print(f"Max mean reward: {np.max(episode_rewards):.2f}")

    if save_video:
        print(f"\n{'='*60}")
        print(f"Video saved to: videos/")
        print(f"Ready to share online!")
        print(f"{'='*60}")


def compare_checkpoints(
    checkpoint_paths,
    env_name="Humanoid-v5",
    num_episodes=10,
    device="mps",
):
    """Compare multiple checkpoints"""

    results = {}

    for checkpoint_path in checkpoint_paths:
        print(f"\nEvaluating: {checkpoint_path}")
        print("-" * 60)

        env = gym.make(env_name)
        env = HumanoidPDWrapper(env, kp=0.5, kd=0.05)
        assert env.observation_space.shape is not None
        assert env.action_space.shape is not None
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        # Create and load agent
        agent = DDPGAgent(obs_size=obs_dim, action_size=action_dim, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.actor.eval()

        episode_rewards = []

        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            num_steps = 0

            for _ in range(1000):
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = agent.actor(state_tensor).cpu().numpy().squeeze()
                action = np.clip(action, -1.0, 1.0)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_reward += float(reward)
                num_steps += 1
                state = next_state

                if done:
                    break

            mean_reward = episode_reward / num_steps if num_steps > 0 else 0.0
            episode_rewards.append(mean_reward)

        env.close()

        avg_mean_reward = np.mean(episode_rewards)
        std_mean_reward = np.std(episode_rewards)
        results[checkpoint_path] = (avg_mean_reward, std_mean_reward)

        print(f"Average mean reward: {avg_mean_reward:.2f} ± {std_mean_reward:.2f}")

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for path, (avg, std) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        print(f"{path}: {avg:.2f} ± {std:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained DDPG agent")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_epfinal.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Humanoid-v5",
        help="Environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device (mps/cuda/cpu)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video of one episode (for sharing online)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Video FPS (frames per second)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple checkpoints",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare mode
        compare_checkpoints(
            checkpoint_paths=args.compare,
            env_name=args.env,
            num_episodes=args.episodes,
            device=args.device,
        )
    else:
        # Visualize mode
        visualize_agent(
            checkpoint_path=args.checkpoint,
            env_name=args.env,
            num_episodes=args.episodes,
            device=args.device,
            save_video=args.save_video,
            video_fps=args.fps,
        )
