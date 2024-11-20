import numpy as np

class TicTacToeEnv:

    def __init__(env):
        env.board = np.zeros((3, 3), dtype=int)  # initialize an empty board
        env.winner = None  
        env.current_player = 1  #player 1 starts the game

    def reset(env):
        env.board = np.zeros((3, 3), dtype=int)
        env.winner = None
        env.current_player = 1
        return env.board  

    def possible_actions(env):
        return [(row, column) for row in range(3) for column in range(3) if env.board[row][column] == 0]
    
    def check_winner(env):
        #check rows
        for row in range(3):
            if abs(sum(env.board[row])) == 3:  
                return True

        # check columns
        for column in range(3):
            if abs(sum(env.board[:, column])) == 3: 
                return True

        # check diagonals
        if abs(sum(env.board[index, index] for index in range(3))) == 3:  # Diagonal 1
            return True
        if abs(sum(env.board[index, 2 - index] for index in range(3))) == 3:  # Diagonal 2
            return True

        return False 

    def is_full(env):
        return not np.any(env.board == 0)

    def move(env, action, player):
 
        row, column = action
        if env.board[row, column] != 0:  
            return env.board, -10  
        
        env.board[row, column] = player
        if env.check_winner():
            return env.board, 1  
        elif env.is_full():
            return env.board, 0  
        
        env.current_player = -env.current_player 
        return env.board, 0  

