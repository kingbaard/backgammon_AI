import configparser
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backgammon_env.envs.game import Game
import pygame

# Set constants
config = configparser.ConfigParser()
config.read('env/envs/config.ini')
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
FPS = 30

class BackgammonEnv(gym.Env):
    """
    ## Description


    ## Action Space
    The action is a `ndarray` with shape `(2,)` which can take integer values from 0 to 25.
    The first element of the array correlates to the board position index of the piece origin.
    The second element of the array describes the destination position index

    ## Observation Space
    {
    0: p1 bar (positive int)
    1-24: number of checkers at each location. (p1 positive, p2 negative)
    25: p2 bar (negative int)
    26: dice1 value (int)
    27: dice2 value (int or None)
    28: dice3 value (int or None)
    29  dice4 value (int or None)
    }

    ## Starting Space
    """
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, p1="ai", p2="ai", render_mode=None,):
        # Game props
        self.p1 = p1
        self.p2 = p2
        self.game = Game(p1, p2)
        

        self.observation_space = spaces.Dict(
            {
                'bar_p2': spaces.Discrete(16),
                'spot_1': spaces.Discrete(16),
                'spot_2': spaces.Discrete(16),
                'spot_3': spaces.Discrete(16),
                'spot_4': spaces.Discrete(16),
                'spot_5': spaces.Discrete(16),
                'spot_6': spaces.Discrete(16),
                'spot_7': spaces.Discrete(16),
                'spot_8': spaces.Discrete(16),
                'spot_9': spaces.Discrete(16),
                'spot_10': spaces.Discrete(16),
                'spot_12': spaces.Discrete(16),
                'spot_13': spaces.Discrete(16),
                'spot_14': spaces.Discrete(16),
                'spot_15': spaces.Discrete(16),
                'spot_16': spaces.Discrete(16),
                'spot_17': spaces.Discrete(16),
                'spot_18': spaces.Discrete(16),
                'spot_19': spaces.Discrete(16),
                'spot_20': spaces.Discrete(16),
                'spot_21': spaces.Discrete(16),
                'spot_22': spaces.Discrete(16),
                'spot_23': spaces.Discrete(16),
                'spot_24': spaces.Discrete(16),
                'bar_p1': spaces.Discrete(16),
                'dice_1': spaces.Discrete(7),
                'dice_2': spaces.Discrete(7),
                'dice_3': spaces.Discrete(7),
                'dice_4': spaces.Discrete(7),
            }
        )
        self.action_space = spaces.Dict({
            'origin': spaces.Discrete(25), #24 spots plus player's bar
            'destination': spaces.Discrete(25), #24 spots plus player's goal
            }
        )

        if self.render_mode == 'human':
            pygame.init()


    def step(self, action):
        origin, destination = action

        # Ensure/assert action is legal, if so change board accordingly
        is_valid = self.game.process_move_ai(action)

        # Get return values 
        observation = self._get_obs()
        reward = self._calculate_reward(is_valid)
        terminated = self._end_check()
        info = self._get_info()

        # TODO: rendering goes here

        return observation, reward, terminated, False, info
    
    def reset(self):
        self.game.start_new_game()
        return self._get_obs()
    
    def close(self):
        if self.render_mode == 'human': 
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        observation_v = np.zeros(29)

        # Add spots on board
        for spot_i, spot in enumerate(self.game.board):
            observation_v[spot_i] = spot.piece_count if spot.player == self.side else spot.piece_count * -1

        # Add dice
        for die_i, die in self.game.dice:
            if die:
                observation_v[25 + die_i] = self.game.dice

        dict = {key: value for key, value in zip(self.observation_space.keys(), observation_v)}
        return dict
    
    def _calculate_reward(self, is_valid, end_status):
        if not is_valid:
            return -0.5
        
        reward = 0
        
        # Reward blocking strategy (+0.1 for every consecutive fortified position this turn)
        
        # Reward offensive plays (+0.1 if you took an enemy piece this turn)

        # Discourage vulnerable pieces (-0.05 for each lonely checker) 

        # Reward winning (+1 if you won this turn)
        if end_status == 'win':
            reward += 10
        # discourage losing (-1 if you lost this turn)
        elif end_status == 'loss':
            reward -= 10

        return reward