import numpy as np

class Agent:
    def __init__(self, side):
        self.side = side

    def _vectorize_observation(self, board, dice):
        observation = np.zeros(29)

        # Add spots on board
        for spot_i, spot in enumerate(board):
            observation[spot_i] = spot.piece_count if spot.player == self.side else spot.piece_count * -1

        # Add dice
        for die_i, die in dice:
            if die:
                observation[25 + die_i] = dice

        return observation
    
    
