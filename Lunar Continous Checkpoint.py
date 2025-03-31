import numpy as np
import torch
import torch.optim as optim
from torch import nn
import gymnasium as gym
import torch.nn.functional as F
import torch.distributions as distributions
from torch.optim.lr_scheduler import StepLR
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)

class PolicyNet(nn.Module):
    def __init__(self, dim_state=8, action_dim=2, net_width=150):
        super().__init__()

        self.l1 = nn.Linear(dim_state, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.alpha_head = nn.Linear(net_width, action_dim)
        self.beta_head = nn.Linear(net_width, action_dim)

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))

        alpha = F.softplus(self.alpha_head(a)) + 1.0
        beta = F.softplus(self.beta_head(a)) + 1.0

        return alpha, beta

class ValueNet(nn.Module):
    def __init__(self, dim_state=8, net_width=150):
        super().__init__()

        self.C1 = nn.Linear(dim_state, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.tanh(self.C1(state))
        v = torch.tanh(self.C2(v))
        v = self.C3(v)
        return v

class PPOAgent:
    def __init__(self, dim_state=8, action_dim=2, lr=3e-4, clip_epsilon=0.2, epochs=20, entropy_coeff=0.1, entropy_coeff_decay=0.99):

        self.policy_net = PolicyNet(dim_state, action_dim).to(device)  # Move model to GPU
        self.value_net = ValueNet(dim_state).to(device)  # Move model to GPU
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.entropy_coeff = entropy_coeff
        self.entropy_coeff_decay = entropy_coeff_decay
        self.policy_net.apply(init_weights)
        self.value_net.apply(init_weights)

        # Learning rate schedulers
        self.policy_scheduler = StepLR(self.policy_optimizer, step_size=100, gamma=0.9)
        self.value_scheduler = StepLR(self.value_optimizer, step_size=100, gamma=0.9)

    def save_checkpoint(self, episode, checkpoint_dir='checkpoints'):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_episode_{episode}.pt')
        torch.save({
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_scheduler_state_dict': self.policy_scheduler.state_dict(),
            'value_scheduler_state_dict': self.value_scheduler.state_dict(),
            'entropy_coeff': self.entropy_coeff,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.policy_scheduler.load_state_dict(checkpoint['policy_scheduler_state_dict'])
        self.value_scheduler.load_state_dict(checkpoint['value_scheduler_state_dict'])
        self.entropy_coeff = checkpoint['entropy_coeff']
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['episode']

    def compute_policy_loss(self, states, actions, old_log_probs, advantages):
        self.entropy_coeff *= self.entropy_coeff_decay
        alpha, beta = self.policy_net(states)
        dist = distributions.Beta(alpha, beta)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        # PPO surrogate loss
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        # Include entropy bonus (note the negative sign to maximize entropy)
        return policy_loss - self.entropy_coeff * entropy

    def compute_value_loss(self, states, returns):
        values = self.value_net(states)
        value_loss = F.mse_loss(values, returns).mean()
        return value_loss

    def update(self, buffer, advantages, returns, batch_size=64):
        # Convert buffer data to tensors and move to device
        states = torch.tensor(np.array(buffer.states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(buffer.actions), dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device).view(-1, 1)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.epochs):
            # Shuffle indices for each epoch
            np.random.shuffle(indices)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Sample mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Compute policy loss and update policy network
                policy_loss = self.compute_policy_loss(batch_states, batch_actions, batch_old_log_probs, batch_advantages)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Compute value loss and update value network
                value_loss = self.compute_value_loss(batch_states, batch_returns)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

        self.policy_scheduler.step()
        self.value_scheduler.step()

        return policy_loss.item(), value_loss.item()

class Buffer:
    def __init__(self):
        self.actions = []
        self.values = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.actions.clear()
        self.values.clear()
        self.states.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

def train(env, test_env, max_episodes=1000, max_steps=300, gamma=0.99, lam=0.95, lr=3e-4, clip_epsilon=0.2, entropy_coeff=0.05, entropy_coeff_decay=0.99, checkpoint_interval=100):
    agent = PPOAgent(dim_state=env.observation_space.shape[0], action_dim=2, lr=lr, clip_epsilon=clip_epsilon, epochs=10, entropy_coeff=entropy_coeff, entropy_coeff_decay=entropy_coeff_decay)
    buffer = Buffer()
    train_rewards = []
    time = 0
    start_episode = 0

    # Load checkpoint if exists
    checkpoint_dir = 'checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_episode_')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            start_episode = agent.load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint)) + 1

    for episode in range(start_episode, max_episodes):
        state, _ = env.reset()
        buffer.clear()
        episode_reward = 0
        step = 0
        done_tr = False
        done = False
        while not done_tr:  # and step<max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Move to device
            with torch.no_grad():
                alpha, beta = agent.policy_net(state_tensor)
                dist = distributions.Beta(alpha, beta)
                a = dist.sample()
                log_prob = dist.log_prob(a).sum(dim=-1)
                action = 2 * (torch.clamp(a, 0, 1) - 0.5)
                value = agent.value_net(state_tensor).item()

            next_state, reward, done, truncated, info = env.step(action.squeeze(0).numpy())
            time += 1
            buffer.states.append(state)
            buffer.values.append(value)
            buffer.actions.append(a.squeeze(0).numpy())
            buffer.log_probs.append(log_prob.item())
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            done_tr = (done or truncated)
            state = next_state
            step += 1
            episode_reward += reward

        train_rewards.append(episode_reward)
        mean_train_rewards = np.mean(train_rewards[-25:])

        # Compute advantages and returns using GAE
        with torch.no_grad():
            final_value = 0 if done else agent.value_net(
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            ).item()
            values = np.array(buffer.values)
            next_values = np.append(values[1:], final_value)

            advantages = []
            gae = 0
            for t in reversed(range(len(buffer.rewards))):
                delta = buffer.rewards[t] + gamma * next_values[t] * (1 - buffer.dones[t]) - values[t]
                gae = delta + gamma * lam * (1 - buffer.dones[t]) * gae
                advantages.insert(0, gae)
            advantages = np.array(advantages)

            returns_array = advantages + values
            # Normalize advantages and returns
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            # returns_array = (returns_array - returns_array.mean()) / (returns_array.std() + 1e-8)

        policy_loss, value_loss = agent.update(buffer, advantages, returns_array)

        if episode % 10 == 0:
            print(f'| Episode: {episode:3} | Time: {int(time / 1000)}K Mean Train Rewards: {mean_train_rewards:7.1f} | '
                  f'Policy Loss: {policy_loss:7.4f} | Value Loss: {value_loss:7.4f} ')
            with open("training_rewards.txt", "a") as f:
                f.write(f"{episode},{int(time / 1000)},{mean_train_rewards}\n")

        if episode % 60 == 0:
            evaluate(test_env, agent)

        # Save checkpoint at regular intervals
        if episode % checkpoint_interval == 0:
            agent.save_checkpoint(episode)

    return agent

def evaluate(env, agent):
    agent.policy_net.eval()
    agent.value_net.eval()
    rewards = []
    done = False
    state = env.reset()[0]
    step = 0
    episode_reward = 0
    while not done and step < 500:
        step = step + 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Move to device
        with torch.no_grad():
            alpha, beta = agent.policy_net(state)
            dist = distributions.Beta(alpha, beta)
            a = dist.sample()
            action = 2 * (torch.clamp(a, 0, 1) - 0.5)

        state, reward, done, truncated, info = env.step(action.squeeze(0).numpy())
        episode_reward += reward

    return episode_reward

if __name__ == '__main__':
    env = gym.make('LunarLander-v3', continuous=True)
    test_env = gym.make('LunarLander-v3', continuous=True, render_mode="human")

    trained_agent = train(env, test_env, max_episodes=10000, max_steps=2000, gamma=0.99, lam=0.95, lr=0.0003, clip_epsilon=0.2, entropy_coeff=0.0, entropy_coeff_decay=0.99, checkpoint_interval=100)