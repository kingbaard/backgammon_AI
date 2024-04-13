import configparser
from colors import BLACK, WHITE, DARK_RED, LIGHT_GREEN, LIGHT_BROWN 
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# Set constants
config = configparser.ConfigParser()
config.read('env/envs/config.ini')
SCREEN_WIDTH = int(config["RENDERING"]["SCREEN_WIDTH"])
SCREEN_HEIGHT = int(config["RENDERING"]["SCREEN_HEIGHT"])
FPS = int(config["RENDERING"]["FPS"])

class BackgammonEnv(gym.Env):
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

    """
    Rendering
    """
    def render(self):
        if self.render_mode == 'human' and not self.screen:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Backgammon Boiis Backgammon")
            self.screen.fill(WHITE)
            self.clock = pygame.time.Clock()
        self.render_board()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def render_board(self):
        board_width = SCREEN_WIDTH - 50
        board_height = 2 * SCREEN_HEIGHT // 3
        board_rect = pygame.Rect(25, 25, board_width, board_height)
        pygame.draw.rect(self.screen, LIGHT_GREEN, board_rect)

        # Draw traiangles
        triangle_width = board_width // 16
        triangle_height = board_height // 2 - 25
        for i in range(6):
            # Triangles on the left side
            left_x = 25 + i * triangle_width
            right_x = 25 + (i + 1) * triangle_width
            color = DARK_RED if i % 2 == 0 else BLACK  # Alternate colors
            
            # Top triangles
            pygame.draw.polygon(self.screen, color, [(left_x, 25), (right_x, 25), (left_x + triangle_width // 2, triangle_height)])
            # Bottom triangles
            pygame.draw.polygon(self.screen, color, [(left_x, board_height + 25), (right_x, board_height + 25), (left_x + triangle_width // 2, board_height + 25 - triangle_height)])
            
            # Triangles on the right side
            left_x = board_width // 2 + i * triangle_width
            right_x = board_width // 2 + (i + 1) * triangle_width
            
            # Top triangles
            pygame.draw.polygon(self.screen, color, [(left_x, 25), (right_x, 25), (left_x + triangle_width // 2, triangle_height)])
            # Bottom triangles
            pygame.draw.polygon(self.screen, color, [(left_x, board_height + 25), (right_x, board_height + 25), (left_x + triangle_width // 2, board_height + 25 - triangle_height)])

    def render_checkers(self):
        
#Test 
test = BackgammonEnv()

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            exit()
    test.render()