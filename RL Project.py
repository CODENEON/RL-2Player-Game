import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# Game Environment
class RPSGameEnv:
    def __init__(self):
        self.actions = ['rock', 'paper', 'scissors']
        self.history = []
        self.scores = [0, 0]  # [Player 1 score, Player 2 score]
        self.streaks = [0, 0]  # [Player 1 streak, Player 2 streak]

    def reset(self):
        self.history = []
        self.scores = [0, 0]
        self.streaks = [0, 0]
        return self.get_state()

    def get_state(self):
        return {
            "history": self.history[-5:],  # Last 5 moves
            "scores": self.scores,
            "streaks": self.streaks
        }

    def step(self, action1, action2):
        self.history.append((action1, action2))
        winner = self.get_winner(action1, action2)

        if winner == 1:
            self.scores[0] += 1
            self.streaks[0] += 1
            self.streaks[1] = 0
        elif winner == 2:
            self.scores[1] += 1
            self.streaks[1] += 1
            self.streaks[0] = 0
        else:
            self.streaks = [0, 0]  # Reset streaks on a draw

        # Add streak bonuses
        if self.streaks[0] >= 3:
            self.scores[0] += 2  # Bonus for Player 1
        if self.streaks[1] >= 3:
            self.scores[1] += 2  # Bonus for Player 2

        return self.get_state(), self.scores

    def get_winner(self, action1, action2):
        if action1 == action2:
            return 0
        elif (action1 == 'rock' and action2 == 'scissors') or \
             (action1 == 'paper' and action2 == 'rock') or \
             (action1 == 'scissors' and action2 == 'paper'):
            return 1
        else:
            return 2

    def render(self):
        print(f"History: {self.history}")
        print(f"Scores: {self.scores}")
        print(f"Streaks: {self.streaks}")

# Minimax Agent
class MinimaxAgent:
    def __init__(self, actions):
        self.actions = actions

    def evaluate(self, state, my_action, opp_action):
        if my_action == opp_action:
            return 0  # Draw
        elif (my_action == 'rock' and opp_action == 'scissors') or \
             (my_action == 'paper' and opp_action == 'rock') or \
             (my_action == 'scissors' and opp_action == 'paper'):
            return 1  # Win
        else:
            return -1  # Loss

    def choose_action(self, history):
        if not history:
            return random.choice(self.actions)

        # Predict the opponent's next move based on history
        opponent_last_move = history[-1][1] if history else random.choice(self.actions)
        best_action = None
        best_score = float('-inf')

        for my_action in self.actions:
            score = self.evaluate(None, my_action, opponent_last_move)
            if score > best_score:
                best_score = score
                best_action = my_action

        return best_action

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)[0]).item()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target

            optimizer.zero_grad()
            output = self.model(state_tensor)
            loss = criterion(output, target_f)
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Visualization
def visualize_results(episodes, minimax_scores, dqn_scores):
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, minimax_scores, label='Minimax', marker='o', linestyle='-', color='b')
    plt.plot(episodes, dqn_scores, label='DQN', marker='x', linestyle='--', color='r')
    plt.title('Scores over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig("line_plot_scores.png")
    plt.show()

# Training and Evaluation
if __name__ == "__main__":
    env = RPSGameEnv()
    minimax_agent = MinimaxAgent(env.actions)
    dqn_agent = DQNAgent(state_size=5, action_size=3)  # State size is history length

    episodes = 150
    batch_size = 32
    minimax_scores = []
    dqn_scores = []

    for episode in range(episodes):
        state = env.reset()
        total_rewards = [0, 0]

        for _ in range(10):  # 10 rounds per game
            action1 = minimax_agent.choose_action(state['history'])
            action2 = dqn_agent.act([0] * 5)  # Placeholder for state
            next_state, scores = env.step(action1, env.actions[action2])

            reward = scores[1] - scores[0]  # Reward for Player 2 (DQN)
            dqn_agent.remember([0] * 5, action2, reward, [0] * 5, False)
            total_rewards[0] += scores[0]
            total_rewards[1] += scores[1]

            if len(dqn_agent.memory) > batch_size:
                dqn_agent.replay(batch_size)

        minimax_scores.append(total_rewards[0])
        dqn_scores.append(total_rewards[1])
        print(f"Episode {episode + 1}: Scores - Minimax: {total_rewards[0]}, DQN: {total_rewards[1]}")

    visualize_results(range(1, episodes + 1), minimax_scores, dqn_scores)
