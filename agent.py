import numpy as np
import random

class TicTacToeAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def get_state_key(self, board):
        """Convert the board to a tuple key for Q-table."""
        return tuple(board.flatten())

    def choose_action(self, board, possible_actions):
        """Choose an action using an epsilon-greedy policy."""
        state_key = self.get_state_key(board)
        if random.uniform(0, 1) < self.epsilon or state_key not in self.q_table:
            return random.choice(possible_actions)
        # Exploit: Choose the action with the highest Q-value
        q_values = self.q_table[state_key]
        return max(q_values, key=q_values.get)

    def update_q_table(self, board, action, reward, next_board, next_possible_actions):
        """Update Q-values using the Q-learning update rule."""
        state_key = self.get_state_key(board)
        next_state_key = self.get_state_key(next_board)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in next_possible_actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0 for a in next_possible_actions}

        current_q = self.q_table[state_key].get(action, 0)
        future_q = max(self.q_table[next_state_key].values(), default=0)
        self.q_table[state_key][action] = current_q + self.alpha * (reward + self.gamma * future_q - current_q)
