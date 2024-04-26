from copy import copy, deepcopy
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers, conversions
from supersuit import black_death_v3
import numpy as np
from backgammon_env.envs.game import Game
import pygame

# Set constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
FPS = 30

def env(**kwargs):
    env = raw_env(**kwargs)
    # env = conversions.aec_to_parallel_wrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env

class raw_env(AECEnv):
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
    metadata = {
        "name": "backgammon_env",
        "render_modes": ["human", "stout"],
        "render_fps": FPS,
        "is_parallelizable": True
        }

    def __init__(self, render_mode=None,):
        # Game props
        self.turn_limit = 500
        self.current_turn = 0
        self.render_mode = render_mode
        self.possible_agents = [0, 1]
        self.game = Game(self.possible_agents[0], self.possible_agents[1])
        self.win_status = None

        low_bounds = np.zeros(30)
        high_bounds = np.zeros(30)
        for i in range(30):
            if i in [0, 25]:
                low = 0
                high = 15
            #Dice
            elif i in [26, 27, 28, 29]:
                low = 1
                high = 6
            else:
                low = -15
                high = 15
            low_bounds[i] = low
            high_bounds[i] = high

        self._observation_spaces = {
            agent: spaces.Dict({
                'observation': spaces.Box(low=low_bounds, high=high_bounds, dtype= np.int8),
                'action_mask': spaces.Box(
                    # 26 origins, 26 destinations
                    # NOTE: 26*26 is a larger action space than actually needed
                    # TODO: Find the smallest possible action space for backgammon
                    low=0, high=1, shape=(26*26,), dtype=np.int8
                ),
            })
            for agent in self.possible_agents
        }
        self.observation_spaces = self._observation_spaces

        self._action_spaces = {agent: spaces.Discrete(676) for agent in self.possible_agents} #25 * 25
        if self.render_mode == 'human':
            pygame.init()
        self.action_spaces = self._action_spaces

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None) #might need to remove this bc of action space masking
    def action_space(self, agent):
        return self._action_spaces[agent]

    def step(self, action):
        # TODO: rendering 
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "stout":
        # else:
            print(f"Current Turn: {self.current_turn}")
            print("\n"+ str(self.game) +"\n")
        
        self.rewards = {a: 0 for a in self.possible_agents}
        # should_end_turn = self.game._end_turn_check()
        agent = self.agent_selection

        if (self.game.turn > self.turn_limit):
            self.terminations = {i: True for i in self.agents}
            self.truncations = {i: True for i in self.agents}
        if (self.terminations[agent] or self.truncations[agent]):
            action = None
            return self._was_dead_step(action)
        
        if not self.game.win_status:
            self.win_status = self.game.win_check()
        if self.terminations[agent]:
            self.rewards = self._calculate_reward()
            self._accumulate_rewards()
            # print(f"\nP0 Rewards: {self._cumulative_rewards[0]}\nP1 Rewards: {self._cumulative_rewards[1]}\n")

        if self.win_status in [1, 0, 2]:
            self.terminations[agent] = True
            self.agent_selection = self._agent_selector.next()
            return 
        if action is None:
            self.observations[agent] = self.observe(agent)
            self.agent_selection = self._agent_selector.next()
            self.rewards = self._calculate_reward()
            self._accumulate_rewards()
            return
        self.state[agent] = action
        unflat_action = self._unflatten_action(action)
        if int(agent) == self.game.current_player.id and action:
            if not self.game.legal_moves:
                self.game.end_turn()
                self.rewards = self._calculate_reward()
                self._accumulate_rewards()
                return
            # if unflat_action not in self.game.legal_moves:
            #     print(f"env.step: {agent} selected an invalid move!")
            #     return
            
            if agent == 1:
                action_full = self._translate_p1_action(unflat_action)
                action = self._flatten_action(*action_full)
            
            readable_action = self._unflatten_action(action)
            # print(f'Agent {agent} is attempting the following move: {readable_action} ({action})')
            # Ensure/assert action is legal, if so change board accordingly
            self.game.process_move_ai(self._unflatten_action(action))

        self.infos[agent] = self._get_infos()
        self.observations[agent] = self.observe(agent)
        self.win_status = self.game.win_check()
        self.rewards = self._calculate_reward()
        self._accumulate_rewards()
        # print(f"\nP0 Rewards: {self.rewards[0]}\nP1 Rewards: {self.rewards[1]}\n")
        # print(f"\nP0 Cul Rewards: {self._cumulative_rewards[0]}\nP1 Cul Rewards: {self._cumulative_rewards[1]}\n")
        if self.game._end_turn_check(agent):
            self.agent_selection = self._agent_selector.next()
    
    def reset(self, seed = None, options = None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: self._get_infos() for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.game.start_new_game()
    
    def close(self):
        if self.render_mode == 'human': 
            pygame.display.quit()
            pygame.quit()

    def render(self):
        pass

    def observation_space(self, agent):
        return self._observation_spaces[int(agent)]
    
    def action_space(self, agent):
        return self._action_spaces[int(agent)]
    
    def _generate_action_mask(self):
        pass

    def _flatten_action(self, origin, destination):
        return 26 * origin + destination

    def _unflatten_action(self, hash):
        origin = hash // 26
        destination = hash % 26
        return (origin, destination)

    def _get_infos(self):
        return {
            "TimeLimit.truncated" : False
        }

    def observe(self, agent):
        # print(f"\n Start observe() for player{agent}")
        self.game.ready_board_ai(agent)
        # observation_dict = {key: 0 for key in self._observation_spaces[agent]['observation'].keys()}
        observation_array = np.zeros(30).astype(np.int8)
        current_agent = self.possible_agents.index(agent)
        board = self.game.board

        # Add spots on board
        for spot_i, spot in enumerate(board):
            if spot.player and spot_i not in [0, 25]:
                # observation_dict[f'spot_{spot_i}'] = spot.piece_count if spot.player.id == self.agent_selection else spot.piece_count + 15
                observation_array[spot_i] = spot.piece_count if spot.player.id == self.agent_selection else spot.piece_count * -1
        # observation_dict['bar_p1'] = board[0].piece_count
        # observation_dict['bar_p2'] = board[25].piece_count
        observation_array[0] = board[0].piece_count
        observation_array[25] = board[25].piece_count
        for die_i, die in enumerate(self.game.dice):
            if die:
                # observation_dict[f'dice_{die_i + 1}'] = self.game.dice[die_i]
                observation_array[die_i + 26] = self.game.dice[die_i]
        
        # If agent 1, flip board values
        if current_agent == 1:
            observation_array = self._translate_p1_observation(observation_array)    
        
        # Sort to ensure consistant item order
        # sorted_keys = sorted(observation_dict.keys())
        # observation_dict = {key: observation_dict[key] for key in sorted_keys}

        # Add action mask
        action_mask = np.zeros(26*26).astype(np.int8)
        if agent == self.agent_selection:
            if agent == 0:
                legal_moves_full = [
                    (origin, destination) for origin, destination in self.game.legal_moves if self.game.dice
                    ]
            else:
                legal_moves_full = [
                    (25 - origin, 25 - destination) for origin, destination in self.game.legal_moves if self.game.dice
                    ]
        else:
            legal_moves_full = []
            
        # Flatten legal moves for model
        legal_moves_flat = [
            self._flatten_action(origin, destination) for origin, destination in legal_moves_full
            ]
        
        for move in legal_moves_flat:
            action_mask[move] = np.int8(1)
        
        return {'observation': observation_array, 'action_mask': action_mask}
    
    def get_action_mask(self):
        if self.observations[self.agent_selection] is None:
            return np.zeros(26*26).astype(np.int8)
        return self.observations[self.agent_selection]['action_mask']
    
    """
    translate p1 agent observation for consistent policy learning. 
    Flips board values.

    accepts:
        Dict: the observation (action_mask excluded)

    returns:
        Dict: the observation as it would appear with board indexes flipped.
    """
    def _translate_p1_observation(self, old_array):
        # assert len(old_dict) > 2, 'This function accepts the observations only'
        # Switch bar space indexes 
        # TODO: find more elegant way of doing this
        # new_dict = copy(old_dict)
        new_array = copy(old_array)
        # new_dict['bar_p1'] = old_dict['bar_p2']
        # new_dict['bar_p2'] = old_dict['bar_p1']

        for i in range(30):
            # if i >=1 and i <=24:
                # new_dict[f'spot_{i}'] = old_dict[f'spot_{25-i}']
            new_array[i] = old_array[30-i-1]
        return new_array

    """
    Translates unflattened action to a normalized board for p1 training consistency.
    Accepts:
        Tuple(int, int): origin, destination unflatten action.
    Returns:
        tuple(int, int): Unflattened action in fliped board perspective 
    """
    def _translate_p1_action(self, unflat_action):
        return (25 - unflat_action[0], 25 - unflat_action[1])

    def _end_check(self, is_valid):
        if self.game.win_status or self.game.turn >= 300 or not is_valid:
            # print(f"is_valid: {is_valid}")
            return True
        return False
    
    def _calculate_reward(self):
        rewards = {0: 0, 1:0}
        
        # Reward blocking strategy (+0.1 for every consecutive fortified position this turn
        for p in rewards.keys():
            longest_block = 0
            current_block = 0
            for position in self.game.board:
                # Discourage vulnerable pieces (1 for each lonely checker)
                if position.player:
                    if position.player.id == p and position.piece_count == 1:
                        rewards[p] -= 1
                    if position.player.id == p and position.piece_count > 1 and position.index not in [0, 25]:
                        current_block += 1
                    else:
                        current_block = 0
                    if current_block > longest_block:
                        longest_block = current_block
            rewards[p] += longest_block * 2
        # Reward number of checkers on opponent's bar
        rewards[0] += self.game.board[25].piece_count * 1
        
        # Reward winning (+10 if you won this turn)
        if self.win_status == 0:
            rewards[0] += 100
            rewards[1] -= 100
        elif self.win_status == 1:
            rewards[0] -= 100
            rewards[1] += 100

        return rewards