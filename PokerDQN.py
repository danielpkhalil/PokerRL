import random
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import texas_holdem_no_limit_v6, texas_holdem_v4
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import uniform_filter1d

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Replay buffer for DQN
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

# DQN agent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, action_mask):
        if np.random.rand() < self.epsilon:
            allowed_actions = np.flatnonzero(action_mask)
            return np.random.choice(allowed_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values.cpu().numpy().flatten()
        q_values[~action_mask.astype(bool)] = -np.inf
        return np.argmax(q_values)

    def train(self, batch_size):
        if len(self.replay_buffer.buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        target_q_values = q_values.clone()
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i, actions[i]] = rewards[i]
            else:
                target_q_values[i, actions[i]] = rewards[i] + self.gamma * torch.max(next_q_values[i]).item()

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Visualization function
def visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy):
    # Apply smoothing to rewards for visualization
    smoothed_agent1_rewards = uniform_filter1d(agent1_rewards, size=50)
    smoothed_agent2_rewards = uniform_filter1d(agent2_rewards, size=50)

    plt.figure(figsize=(16, 10))

    # Plot smoothed rewards
    plt.subplot(3, 1, 1)
    plt.plot(smoothed_agent1_rewards, label="Agent 1 Rewards", color="blue", alpha=0.7)
    plt.plot(smoothed_agent2_rewards, label="Agent 2 Rewards", color="orange", alpha=0.7)
    plt.xlabel("Episodes")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Rewards Over Time")
    plt.legend()
    plt.grid(True)

    # Plot action distributions as stacked bar charts
    plt.subplot(3, 1, 2)
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

    # Plot final total rewards for both agents
    plt.subplot(3, 1, 3)
    final_rewards = [sum(agent1_rewards), sum(agent2_rewards)]
    plt.bar(["Agent 1", "Agent 2"], final_rewards, color=["blue", "orange"], alpha=0.7)
    plt.ylabel("Total Rewards")
    plt.title("Final Total Rewards for Both Agents")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.show()

# Initialize the environment and agents
env = texas_holdem_v4.env(render_mode="none")
env.reset()

# Determine state and action sizes
sample_agent = env.possible_agents[0]
sample_observation, _, _, _, _ = env.last()
state_size = sample_observation["observation"].shape[0]
action_size = env.action_space(sample_agent).n

agent1 = DQNAgent(state_size, action_size)
agent2 = DQNAgent(state_size, action_size)

# Training loop
episodes = 100
batch_size = 32
agents = []
agent1_rewards = []
agent2_rewards = []
agent1_action_counts = np.zeros(action_size)
agent2_action_counts = np.zeros(action_size)

from tqdm import tqdm 
for episode in tqdm(range(episodes)):
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

    # print("episode: ", episode)
    agent1_rewards.append(agent1_total_reward)
    agent2_rewards.append(agent2_total_reward)
    if (episode) % 10 == 0:
        agents.append((agent1, agent2))

# Normalize action counts to represent probabilities
agent1_policy = agent1_action_counts / np.sum(agent1_action_counts)
agent2_policy = agent2_action_counts / np.sum(agent2_action_counts)

# Visualize training results
visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy)


from collections import defaultdict

def is_pocket_pair(cards):
    """
    Detects if a player's cards form a pocket pair with no community cards.

    Args:
        cards: A binary vector of size 52 representing the player's hand and community cards.

    Returns:
        bool: True if the cards represent a pocket pair, False otherwise.
    """
    if sum(cards[:52]) != 2:
        return False  # Ensure only two cards are present (pocket cards)

    # Find the indices of the two cards
    card_indices = [i for i, value in enumerate(cards[:52]) if value == 1]
    if len(card_indices) != 2:
        return False  # Invalid if not exactly 2 cards are present

    # Check if the two cards have the same rank
    card_ranks = [index % 13 for index in card_indices]
    return card_ranks[0] == card_ranks[1]


def load_model(agent, model_path):
    """
    Load a saved model into the agent's BNN.
    """
    agent.model.load_state_dict(torch.load(model_path))
    
    
    
def play_game(agent1, agent2, env):
    """
    Play a single game between two agents in the environment.
    Records winnings for each agent and determines the game winner.

    Returns:
        (float, float): Winnings for agent1 and agent2.
    """
    env.reset()
    agent1_winnings = 0
    agent2_winnings = 0
    pocket_pair_actions = []
    for agent in env.agent_iter():
        observation, reward, done, truncation, info = env.last()
       
                # Accumulate rewards for each agent
        if agent == "player_0":
            agent1_winnings += reward
        else:
            agent2_winnings += reward
            
        if done or truncation:
            env.step(None)
            continue

        state = observation["observation"]
        action_mask = observation["action_mask"]
        cards = state[:52]
        if agent == "player_0":
            action = agent1.act(state, action_mask)
            if is_pocket_pair(cards):
                pocket_pair_actions.append(action)
        else:
            action = agent2.act(state, action_mask)
            if is_pocket_pair(cards):
                pocket_pair_actions.append(action)
        env.step(action)
            
    return agent1_winnings, agent2_winnings, pocket_pair_actions

# Environment setup
env = texas_holdem_v4.env(render_mode="none")
env.reset()

# Test models and record pot sizes
# results = test_models(model_paths, agent_template, env, num_games=10)

# Save results for analysis
def test_best_model(agents, env, test_cases, num_games=10):
    """
    Test the best-trained model against all other model versions.
    
    Args:
        best_model_path: Path to the best-trained model.
        model_paths: List of paths to all model versions.
        agent_template: Function to instantiate an agent.
        env: The game environment.
        num_games: Number of games to play per matchup.
    
    Returns:
        results: Dictionary with winnings and wins for the best model against each other version.
    """
    results = {}

    # Load the best model
    best_agent = agents[-1][0]

    for model in range(len(agents) - 1):

        # Load the opponent model
        opponent_agent = agents[model][0]

        # Track winnings and wins
        best_agent_winnings = 0
        opponent_agent_winnings = 0
        best_agent_wins = 0
        opponent_agent_wins = 0
        all_pocket_pair_actions = []
        for _ in range(num_games):
            # Play the game
            best_winnings, opponent_winnings, pocket_pair_actions = play_game(best_agent, opponent_agent, env)
            all_pocket_pair_actions.extend(pocket_pair_actions)
            # Update winnings
            best_agent_winnings += best_winnings
            opponent_agent_winnings += opponent_winnings

            # Determine game winner
            if best_winnings > opponent_winnings:
                best_agent_wins += 1
            elif opponent_winnings > best_winnings:
                opponent_agent_wins += 1

        # Record results
        results[model] = {
            "best_agent_winnings": best_agent_winnings,
            "opponent_agent_winnings": opponent_agent_winnings,
            "best_agent_wins": best_agent_wins,
            "opponent_agent_wins": opponent_agent_wins,
        
        }

        print(f"Played {num_games} games between best model and {model}")

        plt.hist(all_pocket_pair_actions, bins=range(5), align='left', rwidth=0.8)

# Set x-axis labels
        plt.xticks(ticks=[0, 1, 2, 3], labels=['Call', 'Raise', 'Fold', 'Check'])

        # Add labels and title
        plt.xlabel('Actions')
    
        plt.title('Actions Taken for Pocket Pairs')

        # Display the plot
        plt.show()
    return results

def plot_best_agent_results(results):
    """
    Plot the results of the best agent against all other models.
    
    Args:
        results: Dictionary containing winnings and win counts for the best agent
                 against each opponent model, as produced by `test_best_model`.
    """

    best_agent_winnings = [result["best_agent_winnings"] for result in results.values()]
    expected_winnings = [a/1000 for a in best_agent_winnings]
    opponent_agent_winnings = [result["opponent_agent_winnings"] for result in results.values()]
    best_agent_wins = [result["best_agent_wins"] for result in results.values()]
    opponent_agent_wins = [result["opponent_agent_wins"] for result in results.values()]

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot winnings
    # axes[0].bar(range(len(agents)-1), best_agent_winnings, color='blue', alpha=0.7, label='Best Agent Winnings')
    # axes[0].bar(range(len(agents)-1), opponent_agent_winnings, color='orange', alpha=0.7, label='Opponent Winnings')
    # axes[0].set_ylabel('Total Winnings')
    # axes[0].set_title('Best Agent vs. Other Models - Winnings')
    # axes[0].legend()
    axes[0].scatter([i for i in range(len(expected_winnings))], expected_winnings)
    axes[0].set_title('Expected Best Agent Reward')

    neg = [-1*opponent_agent_wins[i] for i in range(len(opponent_agent_winnings))]
    # Plot win counts
    axes[1].bar(range(len(agents)-1), best_agent_wins, color='blue', alpha=0.7, label='Best Agent Wins')
    axes[1].bar(range(len(agents)-1), neg, color='orange', alpha=0.7, label='Opponent Wins')
    axes[1].set_ylabel('Number of Wins')
    axes[1].set_title('Best Agent vs. Other Models - Win Counts')
    axes[1].legend()

    # Final adjustments
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Model Name')
    plt.tight_layout()
    plt.savefig('figures/best_agent_vs_others_ep_10000')

    # Show the plot
    plt.show()



test_cases = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # game state for AA and no bets
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 7-2 off
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #pair of kings
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # king queen suited
              ]

results = test_best_model(agents, env, test_cases, num_games=1000)
print(results)
plot_best_agent_results(results)