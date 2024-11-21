import numpy as np
import random

class TicTacToeAgent:

    def __init__(agent, epsilon=0.1, alpha=0.5, gamma=0.9):
        
        agent.epsilon = epsilon
        agent.alpha = alpha
        agent.gamma = gamma
        agent.q_table = {}  #qtable to store Qvalues for each state action pair
    
    def _get_state_key(agent, board):
        return tuple(map(tuple, board))  
    
    def choose_action(agent, board, possible_actions):
        if random.random() < agent.epsilon:  #choose a random action
            return random.choice(possible_actions)
        else:  #choose the best action based on the qtable
            state_key = agent._get_state_key(board)
            if state_key not in agent.q_table:
                agent.q_table[state_key] = {action: 0 for action in possible_actions}
            return max(agent.q_table[state_key], key=agent.q_table[state_key].get)
    
    def update_q_table(self, board, action, reward, next_board, possible_actions):
        state_key = self._get_state_key(board)
        next_state_key = self._get_state_key(next_board)

        #initialize qvalues if not already present
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0 for action in possible_actions}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {action: 0 for action in possible_actions}
        
        #get the current Qvalue for the state action pair
        old_q_value = self.q_table[state_key].get(action, 0)
        
        #get the maximum qvalue for the next state 
        future_q_value = max(self.q_table[next_state_key].values())
        
        # Update the qvalue using the Qlearning update rule
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * future_q_value - old_q_value)
        self.q_table[state_key][action] = new_q_value
