import random
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import texas_holdem_no_limit_v6
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d
import pyro
from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
from typing import Sequence


class BNN(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()

        self.activation = nn.Tanh()  # Could also be ReLU or LeakyReLU
        # assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0

        # Define the layer sizes and PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx])
            for idx in range(1, len(self.layer_sizes))
        ]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        # Define priors for each layer
        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                    [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]
                ).to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1)
            )

    def forward(self, x, y=None):
        x = x.reshape(-1, self.layer_sizes[0])  # Ensure input shape matches expected dimensions
        for layer in self.layers[:-1]:  # Apply activation between all layers except the last
            x = self.activation(layer(x))
        mu = self.layers[-1](x).squeeze(-1)  # Final layer without activation

        # Define posterior predictive distribution
        sigma = pyro.sample("sigma", dist.Gamma(2., 1.))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
        return mu


class BNNRegressor(PyroModule):
    def __init__(self, dims: Sequence[int]):
        super().__init__()
        assert dims[-1] == 1
        self.bnn = BNN(dims)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        mu = self.bnn(x).squeeze()
        sigma = pyro.sample("sigma", dist.Uniform(0, 0.5))
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
        return mu


# Replay buffer for BNN-based Bayesian agent
class ReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = []
        self.max_size = size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Bayesian Agent
class BayesianAgent:
    def __init__(self, state_size, action_size, history_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.history_size = history_size

        # History of played cards
        self.history = deque(maxlen=self.history_size)

        # Define a BNN model for each action
        self.models = [BNN(in_dim=state_size + self.history_size, out_dim=1, hid_dim=128, n_hid_layers=3) for _ in range(action_size)]
        self.optimizers = [Adam({"lr": 0.001}) for _ in range(action_size)]
        self.svis = [
            SVI(model=model, guide=pyro.infer.autoguide.AutoDiagonalNormal(model), optim=opt, loss=Trace_ELBO())
            for model, opt in zip(self.models, self.optimizers)
        ]

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def update_history(self, cards):
        """Update the history with newly played cards."""
        self.history.extend(cards)

    def get_augmented_state(self, state):
        """Augment the state with the history of played cards."""
        history_array = np.array(self.history).flatten()
        padded_history = np.pad(history_array, (0, max(0, self.history_size - len(history_array))), mode='constant')
        return np.concatenate([state, padded_history])

    def act(self, state, action_mask):
        """
        Select an action using epsilon-greedy policy.
        """
        augmented_state = self.get_augmented_state(state)
        state_tensor = torch.FloatTensor(augmented_state).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            allowed_actions = np.flatnonzero(action_mask)
            return np.random.choice(allowed_actions)

        # Exploitation: Evaluate actions using the BNNs
        q_values = []
        for action, model in enumerate(self.models):
            if action_mask[action]:
                with torch.no_grad():
                    q_value = model(state_tensor).item()
                q_values.append(q_value)
            else:
                q_values.append(float("-inf"))  # Mask invalid actions

        return np.argmax(q_values)

    def train(self, batch_size):
        """
        Train the BNN models using replayed experiences.
        """
        if len(self.replay_buffer.buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)

        # Train each model separately on its respective data
        for action in range(self.action_size):
            action_indices = [i for i, a in enumerate(actions) if a == action]
            if len(action_indices) == 0:
                continue  # Skip if no examples for this action

            action_states = states[action_indices]
            action_rewards = rewards[action_indices]

            svi = self.svis[action]
            svi.step(action_states, action_rewards)

        # Decay epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Visualization function
def visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy):
    # Apply smoothing to rewards for visualization
    smoothed_agent1_rewards = uniform_filter1d(agent1_rewards, size=50)
    smoothed_agent2_rewards = uniform_filter1d(agent2_rewards, size=50)

    plt.figure(figsize=(14, 8))

    # Plot smoothed rewards
    plt.subplot(2, 1, 1)
    plt.plot(smoothed_agent1_rewards, label="Agent 1 Rewards", color="blue", alpha=0.7)
    plt.plot(smoothed_agent2_rewards, label="Agent 2 Rewards", color="orange", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Rewards Over Time")
    plt.legend()
    plt.grid(True)

    # Plot action distributions as stacked bar charts
    plt.subplot(2, 1, 2)
    bar_width = 0.35
    x = np.arange(len(agent1_policy))
    plt.bar(x - bar_width / 2, agent1_policy, bar_width, label="Agent 1", color="blue", alpha=0.7)
    plt.bar(x + bar_width / 2, agent2_policy, bar_width, label="Agent 2", color="orange", alpha=0.7)
    plt.xlabel("Actions")
    plt.ylabel("Action Selection Frequency")
    plt.title("Action Selection Policies")
    plt.xticks(range(len(agent1_policy)))
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_posterior(y_samples: np.ndarray, f_samples: np.ndarray = None) -> None:
    """
    Plots samples from the posterior, along with ±1 std. uncertainty bands.
    y = f + eps

    Args:
        y_samples: Samples of y from the posterior (n_samples x n_points).
        f_samples: Samples of f(x) from the posterior, optional (n_samples x n_points).
    """
    # Compute statistics for y_samples
    pred_mean = y_samples.mean(axis=0)
    pred_std = y_samples.std(axis=0)
    pred_var = pred_std ** 2
    print('Avg. predictive uncertainty:', pred_var.mean())

    if f_samples is not None:
        # Compute statistics for f_samples
        model_mean = f_samples.mean(axis=0)
        model_std = f_samples.std(axis=0)
        model_var = model_std ** 2
        print('Avg. model uncertainty:', model_var.mean())

        # Compute data uncertainty
        data_var = np.abs(pred_var - model_var)
        data_std = np.sqrt(data_var)
        print('Avg. data uncertainty:', data_var.mean())

    # Plot the results
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
    x = np.arange(y_samples.shape[1])  # Assuming x-coordinates are indices

    ax.plot(x, pred_mean, c='blue', label='Predictive Mean')

    # Predictive uncertainty band
    ax.fill_between(x, pred_mean + pred_std, pred_mean - pred_std, 
                    alpha=0.5, color='blue', label='Predictive Uncertainty')

    if f_samples is not None:
        # Model uncertainty band
        ax.fill_between(x, model_mean + model_std, model_mean - model_std, 
                        alpha=0.5, color='orange', label='Model Uncertainty')

        # Data uncertainty band
        ax.fill_between(x, model_mean + model_std, model_mean + pred_std, 
                        color='red', alpha=0.5, label='Data Uncertainty')
        ax.fill_between(x, model_mean - model_std, model_mean - pred_std, 
                        color='red', alpha=0.5)

    ax.legend()
    plt.xlabel("Index (x)")
    plt.ylabel("Output (y)")
    plt.title("Posterior Predictive with Uncertainty Bands")
    plt.grid(True)
    plt.show()

# Initialize the environment and agents
env = texas_holdem_no_limit_v6.env()
env.reset()

# Determine state and action sizes
sample_agent = env.possible_agents[0]
sample_observation, _, _, _, _ = env.last()
state_size = sample_observation["observation"].shape[0]
action_size = env.action_space(sample_agent).n

# Include history size
history_size = 10

agent1 = BayesianAgent(state_size, action_size, history_size)
agent2 = BayesianAgent(state_size, action_size, history_size)

# Training loop
episodes = 100
batch_size = 32
agent1_rewards = []
agent2_rewards = []
agent1_action_counts = np.zeros(action_size)
agent2_action_counts = np.zeros(action_size)

for episode in range(episodes):
    env.reset()
    agent1_total_reward = 0
    agent2_total_reward = 0

    for agent in env.agent_iter():
        observation, reward, done, truncation, info = env.last()
        if agent == "player_0":
            agent1_total_reward += reward
        else:
            agent2_total_reward += reward

        if done or truncation:
            env.step(None)
            continue

        state = observation["observation"]
        action_mask = observation["action_mask"]

        if agent == "player_0":
            action = agent1.act(state, action_mask)
            agent1_action_counts[action] += 1
            agent1.update_history(observation.get("played_cards", []))
        else:
            action = agent2.act(state, action_mask)
            agent2_action_counts[action] += 1
            agent2.update_history(observation.get("played_cards", []))

        env.step(action)
        next_observation, reward, done, truncation, _ = env.last()
        next_state = next_observation["observation"]

        if agent == "player_0":
            agent1.replay_buffer.add((agent1.get_augmented_state(state), action, reward, agent1.get_augmented_state(next_state), done))
            agent1.train(batch_size)
        else:
            agent2.replay_buffer.add((agent2.get_augmented_state(state), action, reward, agent2.get_augmented_state(next_state), done))
            agent2.train(batch_size)

    agent1_rewards.append(agent1_total_reward)
    agent2_rewards.append(agent2_total_reward)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{episodes} completed.")

# Normalize action counts to represent probabilities
agent1_policy = agent1_action_counts / np.sum(agent1_action_counts)
agent2_policy = agent2_action_counts / np.sum(agent2_action_counts)

# Generate posterior samples
test_X = np.random.uniform(-1, 1, size=(100, state_size + history_size))  # Adjust input size for test samples
test_X_tensor = torch.FloatTensor(test_X)

# Collect posterior samples
y_samples = []
f_samples = []  # Optional: Collect model uncertainty if applicable

for _ in range(100):  # Number of posterior samples
    with torch.no_grad():
        y_sample = agent1.models[0](test_X_tensor).numpy()  # Predict with BNN
        y_samples.append(y_sample)

# Convert to numpy arrays
y_samples = np.array(y_samples)
f_samples = None  # Optionally, if model predictions separate from noise

# Visualize posterior
plot_posterior(y_samples, f_samples)

# Visualize training results
visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy)
