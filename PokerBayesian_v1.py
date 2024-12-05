import random
import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.classic import texas_holdem_v4
from collections import defaultdict
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
    def __init__(self, in_dim=72, out_dim=5, hid_dim=10, n_hid_layers=5, prior_scale=5.):
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
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        # Define a BNN model for each action
        self.model = BNN(in_dim=state_size, out_dim=4, hid_dim=128, n_hid_layers=3)
        self.optimizer = Adam({"lr": 0.001})
        self.svi = SVI(model=self.model, guide=pyro.infer.autoguide.AutoDiagonalNormal(self.model), optim=self.optimizer, loss=Trace_ELBO())
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
    def act(self, state, action_mask):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state (np.ndarray): The current state.
            action_mask (np.ndarray): Mask indicating valid actions.
        Returns:
            action (int): Selected action index.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Random action among allowed actions
            allowed_actions = np.flatnonzero(action_mask)
            return np.random.choice(allowed_actions)

        # Exploitation: Evaluate actions using the BNN
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).numpy()  # Get Q-values for all actions
            masked_q_values = np.where(action_mask, q_values, float("-inf"))  # Apply the mask
            return np.argmax(masked_q_values)  # Choose the best action

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
            svi = self.svi
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
    
    most_likely_action = y_samples.argmax(axis=2)  # Shape: (n_samples, n_points)

    # Compute statistics for the most likely action
    pred_mean = most_likely_action.mean(axis=0)  # Mean across samples for the most likely action
    pred_std = most_likely_action.std(axis=0)   # Std dev across samples for the most likely action
    pred_var = pred_std ** 2

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
    print("x shape:", x.shape)
    print("pred_mean shape:", pred_mean.shape)
    print("pred_std shape:", pred_std.shape)

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
env = texas_holdem_v4.env(render_mode="none")
env.reset()
# Determine state and action sizes
sample_agent = env.possible_agents[0]
sample_observation, _, _, _, _ = env.last()
state_size = sample_observation["observation"].shape[0]
action_size = env.action_space(sample_agent).n
agent1 = BayesianAgent(state_size, action_size)
agent2 = BayesianAgent(state_size, action_size)
# Training loop
episodes = 10000
num_saved = 2

agents = []

batch_size = 32
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
        
        #print("state: ", state)
        if agent == "player_0":
            agent1.replay_buffer.add((state, action, reward, next_state, done))
            agent1.train(batch_size)
        else:
            agent2.replay_buffer.add((state, action, reward, next_state, done))
            agent2.train(batch_size)
            
    # print("episode: ", episode)
    agent1_rewards.append(agent1_total_reward)
    agent2_rewards.append(agent2_total_reward)
    if (episode) % 1000 == 0:
        agents.append((agent1, agent2))
        # torch.save(agent1.model.state_dict(), f"{save_dir}/agent1_{episode}.pth")
        # torch.save(agent2.model.state_dict(), f"{save_dir}/agent2_{episode}.pth")
        # print(f"Saved models at episode {episode}")
# Normalize action counts to represent probabilities
agent1_policy = agent1_action_counts / np.sum(agent1_action_counts)
agent2_policy = agent2_action_counts / np.sum(agent2_action_counts)
# # Generate posterior samples
test_X = np.random.uniform(-1, 1, size=(100, 54))  # Generate 100 samples with 54 features
test_X_tensor = torch.FloatTensor(test_X)


# # Collect posterior samples
y_samples = []
f_samples = []  # Optional: Collect model uncertainty if applicable

for _ in range(100):  # Number of posterior samples
    with torch.no_grad():
        y_sample = agent1.model(test_X_tensor).numpy()  # Predict with BNN
        y_samples.append(y_sample)

# # Convert to numpy arrays
y_samples = np.array(y_samples)
f_samples = None  # Optionally, if model predictions separate from noise

# # Visualize poste
plot_posterior(y_samples, f_samples)

# # Visualize training results
visualize_training(agent1_rewards, agent2_rewards, agent1_policy, agent2_policy)
from collections import defaultdict

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

        if agent == "player_0":
            action = agent1.act(state, action_mask)
        else:
            action = agent2.act(state, action_mask)

        env.step(action)
            
    return agent1_winnings, agent2_winnings



# def test_models(model_paths, agent_template, env, num_games=10):
#     """
#     Play all saved models against each other and record winnings and wins.

#     Returns:
#         results: Dictionary with winnings and wins for each model pair.
#     """
#     results = defaultdict(lambda: {"agent1_winnings": 0, "agent2_winnings": 0, "agent1_wins": 0, "agent2_wins": 0})
#     num_models = len(model_paths)

#     for i in range(num_models):
#         for j in range(num_models):
#             model1_path = model_paths[i]
#             model2_path = model_paths[j]

#             # Load models
#             agent1 = agent_template()
#             agent2 = agent_template()
#             load_model(agent1, model1_path)
#             load_model(agent2, model2_path)

#             # Play games
#             for _ in range(num_games):
#                 agent1_winnings, agent2_winnings = play_game(agent1, agent2, env)

#                 # Update total winnings
#                 results[(model1_path, model2_path)]["agent1_winnings"] += agent1_winnings
#                 results[(model1_path, model2_path)]["agent2_winnings"] += agent2_winnings

#                 # Determine the game winner
#                 if agent1_winnings > agent2_winnings:
#                     results[(model1_path, model2_path)]["agent1_wins"] += 1
#                 elif agent2_winnings > agent1_winnings:
#                     results[(model1_path, model2_path)]["agent2_wins"] += 1

#             print(f"Played {num_games} games between {os.path.basename(model1_path)} and {os.path.basename(model2_path)}")

#     return results

# Environment setup
env = texas_holdem_v4.env(render_mode="none")
env.reset()

# Define agent template
def agent_template():
    return BayesianAgent(state_size=72, action_size=4)

# Test models and record pot sizes
# results = test_models(model_paths, agent_template, env, num_games=10)

# Save results for analysis
def test_best_model(agents, env, num_games=10):
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

        for _ in range(num_games):
            # Play the game
            best_winnings, opponent_winnings = play_game(best_agent, opponent_agent, env)
            
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

    return results

def plot_best_agent_results(results):
    """
    Plot the results of the best agent against all other models.
    
    Args:
        results: Dictionary containing winnings and win counts for the best agent
                 against each opponent model, as produced by `test_best_model`.
    """

    best_agent_winnings = [result["best_agent_winnings"] for result in results.values()]
    opponent_agent_winnings = [result["opponent_agent_winnings"] for result in results.values()]
    best_agent_wins = [result["best_agent_wins"] for result in results.values()]
    opponent_agent_wins = [result["opponent_agent_wins"] for result in results.values()]

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot winnings
    axes[0].bar(range(len(agents)-1), best_agent_winnings, color='blue', alpha=0.7, label='Best Agent Winnings')
    axes[0].bar(range(len(agents)-1), opponent_agent_winnings, color='orange', alpha=0.7, label='Opponent Winnings')
    axes[0].set_ylabel('Total Winnings')
    axes[0].set_title('Best Agent vs. Other Models - Winnings')
    axes[0].legend()

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

results = test_best_model(agents, env, num_games=1000)
print(results)
plot_best_agent_results(results)