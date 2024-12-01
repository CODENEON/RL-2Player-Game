import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output

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


# Visualization Function
import random
import numpy as np
import matplotlib.pyplot as plt

# Visualization Function
def visualize_live(episodes, minimax_scores, dqn_scores, minimax_wins, dqn_wins):
    plt.clf()
    plt.plot(episodes, minimax_scores, label="Minimax Agent", color="blue")
    plt.plot(episodes, dqn_scores, label="DQN Agent", color="red")
    plt.title("Scores Over Time")
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Annotate winner info at the top
    plt.text(0.5, 0.95, 
             f"Minimax Wins: {minimax_wins} | DQN Wins: {dqn_wins}", 
             ha="center", va="center", transform=plt.gca().transAxes, 
             fontsize=12, color="green")
    plt.pause(0.01)  # Pause for updates


# Main Loop
if __name__ == "__main__":
    env = RPSGameEnv()
    minimax_agent = MinimaxAgent(env.actions)

    episodes = 150
    minimax_scores = []
    dqn_scores = []

    minimax_wins = 0
    dqn_wins = 0

    plt.ion()  # Turn on interactive mode for live updates
    plt.figure(figsize=(10, 5))

    for episode in range(episodes):
        state = env.reset()
        minimax_total = 0
        dqn_total = 0

        for _ in range(10):  # 10 rounds per episode
            action1 = minimax_agent.choose_action(state['history'])
            action2 = random.choice(env.actions)  # Random choice for simplicity
            next_state, scores = env.step(action1, action2)

            minimax_total += scores[0]
            dqn_total += scores[1]

        minimax_scores.append(minimax_total)
        dqn_scores.append(dqn_total)

        # Update wins count
        if minimax_total > dqn_total:
            minimax_wins += 1
        elif dqn_total > minimax_total:
            dqn_wins += 1

        visualize_live(range(1, len(minimax_scores) + 1), minimax_scores, dqn_scores, minimax_wins, dqn_wins)
        print(f"Episode {episode + 1}: Minimax = {minimax_total}, DQN = {dqn_total}")

    # Finalize and Show
    plt.ioff()
    plt.show()

    print(f"Final Results: Minimax Wins = {minimax_wins}, DQN Wins = {dqn_wins}")

