import random
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import texas_holdem_no_limit_v6
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d

# Define a Bayesian Network for decision-making
class BayesianNetwork:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Prior probabilities for actions
        self.action_priors = np.ones(action_size) / action_size
        # Transition probabilities: P(s' | s, a)
        self.transition_probs = defaultdict(lambda: defaultdict(lambda: np.ones(state_size) / state_size))
        # Reward probabilities: P(r | s, a)
        self.reward_probs = defaultdict(lambda: defaultdict(lambda: np.ones(2)))  # Binary rewards (0 or 1)

    def update_transition_probs(self, state, action, next_state):
        """
        Update transition probabilities based on observed transitions.
        """
        self.transition_probs[state][action][next_state] += 1
        self.transition_probs[state][action] /= self.transition_probs[state][action].sum()

    def update_reward_probs(self, state, action, reward):
        """
        Update reward probabilities based on observed rewards.
        """
        self.reward_probs[state][action][reward] += 1
        self.reward_probs[state][action] /= self.reward_probs[state][action].sum()

    def get_action_value(self, state, action, gamma=0.99):
        """
        Compute the expected value of an action given the current state.
        """
        expected_value = 0.0
        for next_state in range(self.state_size):
            transition_prob = self.transition_probs[state][action][next_state]
            reward_prob = self.reward_probs[state][action]
            expected_reward = reward_prob[1]  # Probability of reward = 1
            expected_value += transition_prob * (expected_reward + gamma * self.get_state_value(next_state))
        return expected_value

    def get_state_value(self, state, gamma=0.99):
        """
        Compute the value of a state by considering the best action.
        """
        return max(self.get_action_value(state, action, gamma) for action in range(self.action_size))

# Replay buffer for Bayesian agent
class BayesianReplayBuffer:
    def __init__(self, size=10000):
        self.buffer = []
        self.max_size = size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Bayesian agent class
class BayesianAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.network = BayesianNetwork(state_size, action_size)
        self.replay_buffer = BayesianReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def act(self, state, action_mask):
        """
        Select an action using epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            allowed_actions = np.flatnonzero(action_mask)
            return np.random.choice(allowed_actions)
        q_values = [self.network.get_action_value(state, action) for action in range(self.action_size)]
        q_values = np.array(q_values)
        q_values[~action_mask.astype(bool)] = -np.inf
        return np.argmax(q_values)

    def train(self, batch_size):
        """
        Train the Bayesian Network using replayed experiences.
        """
        if len(self.replay_buffer.buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            self.network.update_transition_probs(state, action, next_state)
            self.network.update_reward_probs(state, action, reward)

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

# Initialize the environment and agents
env = texas_holdem_no_limit_v6.env()
env.reset()

# Determine state and action sizes
sample_agent = env.possible_agents[0]
sample_observation, _, _, _, _ = env.last()
state_size = sample_observation["observation"].shape[0]
action_size = env.action_space(sample_agent).n

agent1 = BayesianAgent(state_size, action_size)
agent2 = BayesianAgent(state_size, action_size)

# Training loop
episodes = 1000
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
        else:
            action = agent2.act(state, action_mask)
            agent2_action_counts[action] += 1

        env.step(action)
        next_observation, reward, done, truncation, _ = env.last()
        next_state = next_observation["observation"]

        if agent == "player_0":
            agent1.replay_buffer.add((state, action, reward, next_state, done))
            agent1.train(batch_size)
        else:
            agent2.replay_buffer.add((state, action, reward, next_state, done))
            agent2.train(batch_size)

    agent1_rewards.append(agent1_total_reward)
    agent2_rewards.append(agent2_total_reward)
    agent1.update_target_model()
    agent2.update_target_model()

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}/{episodes} completed.")

# Normalize action counts to represent probabilities
agent1_policy = agent1_action_counts / np.sum(agent1_action_counts)
agent2_policy = agent2_action_counts / np.sum(agent2_action_counts)

# Visualize training results
visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy)
