import configparser
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from colors import BLACK, WHITE, DARK_RED, LIGHT_GREEN, LIGHT_BROWN 
from game import Game
import pygame

# Set constants
config = configparser.ConfigParser()
config.read('env/envs/config.ini')
SCREEN_WIDTH = int(config["RENDERING"]["SCREEN_WIDTH"])
SCREEN_HEIGHT = int(config["RENDERING"]["SCREEN_HEIGHT"])
FPS = int(config["RENDERING"]["FPS"])

class BackgammonEnv(gym.Env):
    """
    ## Description

    ## Action Space
    The action is a `ndarray` with shape `(2,)` which can take integer values from 0 to 25.
    The first element of the array correlates to the board position index of the piece origin.
    The second element of the array describes the destination position index
    """
    metadata = {"render_modes": ["human"], "render_fps": FPS}

    def __init__(self, render_mode=None):
        self.render_mode = 'human'
        self.screen = None
        self.clock = None
        
        if self.render_mode == 'human':
            pygame.init()

    def step(self):
        raise NotImplemented
    
    def reset(self):
        raise NotImplemented
    
    def close(self):
        raise NotImplemented

        
#Test 
test = BackgammonEnv()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            exit()
    test.render()