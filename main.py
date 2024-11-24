import numpy as np
import random

class TicTacToeAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {}  # Q-table to store Q-values for state-action pairs

    def _get_state_key(self, board):
        return tuple(map(tuple, board))  # Convert board state to a hashable tuple

    def choose_action(self, board, possible_actions):
        state_key = self._get_state_key(board)
        if random.random() < self.epsilon:  # Explore: choose a random action
            return random.choice(possible_actions)
        else:  # Exploit: choose the best action based on Q-table
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0 for action in possible_actions}
            return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update_q_table(self, board, action, reward, next_board, possible_actions):
        state_key = self._get_state_key(board)
        next_state_key = self._get_state_key(next_board)

        # Initialize Q-values if not already present
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in possible_actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0 for action in possible_actions}

        # Get current Q-value
        old_q_value = self.q_table[state_key].get(action, 0)
        # Get the maximum Q-value for the next state
        future_q_value = max(self.q_table[next_state_key].values(), default=0)

        # Update Q-value using Q-learning formula
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)
        self.q_table[state_key][action] = new_q_value

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # Initialize empty board
        self.winner = None
        self.current_player = 1  # Player 1 starts

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.current_player = 1
        return self.board

    def possible_actions(self):
        return [(row, column) for row in range(3) for column in range(3) if self.board[row][column] == 0]

    def check_winner(self):
        for row in range(3):  # Check rows
            if abs(sum(self.board[row])) == 3:
                return True
        for column in range(3):  # Check columns
            if abs(sum(self.board[:, column])) == 3:
                return True
        if abs(sum(self.board[i, i] for i in range(3))) == 3:  # Diagonal 1
            return True
        if abs(sum(self.board[i, 2 - i] for i in range(3))) == 3:  # Diagonal 2
            return True
        return False

    def is_full(self):
        return not np.any(self.board == 0)

    def move(self, action, player):
        row, column = action
        if self.board[row, column] != 0:  # Invalid move
            return self.board, -10

        self.board[row, column] = player
        if self.check_winner():  # Check if the move resulted in a win
            return self.board, 1
        elif self.is_full():  # Check if the board is full
            return self.board, 0

        self.current_player = -self.current_player  # Switch player
        return self.board, 0

def train_agents(agent1, agent2, env, episodes=10000):
    print_interval = 1000  # Interval to print progress

    for episode in range(episodes):
        env.reset()
        board = env.board
        current_agent = agent1
        moves = []

        while True:
            possible_actions = env.possible_actions()
            action = current_agent.choose_action(board, possible_actions)
            next_board, reward = env.move(action, env.current_player)
            next_possible_actions = env.possible_actions()

            current_agent.update_q_table(board, action, reward, next_board, next_possible_actions)

            moves.append((env.current_player, action))

            if reward != 0 or env.is_full():
                if episode < 5 or episode >= episodes - 5:  # Print first and last 5 episodes
                    print(f"Episode {episode + 1}:")
                    print("Final Board:")
                    print(env.board)
                    print(f"Winner: {'Player 1' if reward == 1 else 'Player 2' if reward == -1 else 'Draw'}")
                    print("Moves:", moves)
                    print("-" * 30)
                break

            board = next_board
            current_agent = agent2 if current_agent == agent1 else agent1

        if (episode + 1) % print_interval == 0:
            print(f"Training Progress: Episode {episode + 1}/{episodes}")

    print("Training complete!")

if __name__ == "__main__":
    env = TicTacToeEnv()
    agent1 = TicTacToeAgent(epsilon=0.1, alpha=0.5, gamma=0.9)
    agent2 = TicTacToeAgent(epsilon=0.1, alpha=0.5, gamma=0.9)

    print("Training agents...")
    train_agents(agent1, agent2, env, episodes=10000)
# create one agent play
