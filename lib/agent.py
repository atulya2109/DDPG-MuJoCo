from lib.models import DDPGActor, DDPGCritic, ReplayBuffer
import numpy as np
import torch
import torch.nn.functional as F
import copy


class OUNoise:
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.2, eps=1.0):
        self.size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.eps = eps
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class DDPGAgent:
    def __init__(
        self,
        obs_size,
        action_size,
        device="mps",
        gamma=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-3,
        buffer_capacity=1000000,
        policy_delay=2,
        target_noise=0.2,
        noise_clip=0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_size = action_size
        self.policy_delay = policy_delay
        self.total_it = 0  # Track total training iterations
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        self.actor = DDPGActor(obs_size, action_size).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)

        # Twin Critics (TD3) - prevents overestimation
        self.critic_1 = DDPGCritic(obs_size, action_size).to(device)
        self.critic_2 = DDPGCritic(obs_size, action_size).to(device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, obs_size, action_size)

        # Exploration noise
        self.noise = OUNoise(action_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions = self.actor(state).cpu().numpy().squeeze()

        noise = self.noise.sample()
        actions = actions + noise
        actions = np.clip(actions, -1.0, 1.0)

        return actions

    def reset_noise(self):
        self.noise.reset()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return None, None

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # -------- Update Twin Critics (Every Step) --------
        with torch.no_grad():
            # Target policy smoothing: add noise to target actions
            next_actions = self.target_actor(next_states)
            noise = (torch.randn_like(next_actions) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            # Compute both target Q-values and take MINIMUM (key TD3 trick!)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)  # Clipped double Q-learning
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Update Critic 1
        current_q1 = self.critic_1(states, actions)
        critic_1_loss = F.mse_loss(current_q1, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()

        # Update Critic 2
        current_q2 = self.critic_2(states, actions)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()

        # Average critic loss for logging
        critic_loss = (critic_1_loss.item() + critic_2_loss.item()) / 2

        actor_loss = None

        # -------- Delayed Policy Update --------
        if self.total_it % self.policy_delay == 0:
            # Freeze critics to avoid computing unnecessary gradients
            for param in self.critic_1.parameters():
                param.requires_grad = False
            for param in self.critic_2.parameters():
                param.requires_grad = False

            # Use first critic for policy gradient (standard TD3)
            actor_loss = -self.critic_1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Unfreeze critics
            for param in self.critic_1.parameters():
                param.requires_grad = True
            for param in self.critic_2.parameters():
                param.requires_grad = True

            # Soft update target networks (only when actor updates)
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic_1, self.critic_1)
            self.soft_update(self.target_critic_2, self.critic_2)

            actor_loss = actor_loss.item()

        return actor_loss, critic_loss
