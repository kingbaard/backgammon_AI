from copy import copy, deepcopy
import functools
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers 
import numpy as np
from backgammon_env.envs.game import Game
import pygame

# Set constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 500
FPS = 30

class BackgammonEnv(AECEnv):
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
        "render_modes": ["human"],
        "render_fps": FPS
        }

    def __init__(self, render_mode=None,):
        # Game props
        self.turn_limit = 300
        self.current_turn = 0
        self.render_mode = render_mode
        self.possible_agents = ["0", "1"]
        self.game = Game(self.possible_agents[0], self.possible_agents[1])
        
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    'observation': spaces.Dict(
                        {
                        'bar_p2': spaces.Discrete(16), # Opponent's bar
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
                        'spot_11': spaces.Discrete(16),
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
                    ),
                    'action_mask': spaces.Box(
                        # 26 origins, 26 destinations
                        low=0, high=1, shape=(26*26,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self._action_spaces = {agent: spaces.Discrete(676) for agent in self.possible_agents} #25 * 25

        if self.render_mode == 'human':
            pygame.init()

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    # @functools.lru_cache(maxsize=None) #might need to remove this bc of action space masking
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        return np.array(self.observations[agent])

    def step(self, action):
        agent = self.agent_selection
        self.win_status = self.game.win_check()
        self.rewards[agent] = self._calculate_reward(agent, self.game.win_status)

        if (self.terminations[agent]):
            return self._was_dead_step(action)
        if action is None:
            self.observations[agent] = self.observe(agent)
            self.agent_selection = self._agent_selector.next()
            return
        
        self.state[agent] = action
        unflat_action = self._unflatten_action(action)
        is_valid = False
        if int(agent) == self.game.current_player.id and action:
            if not self.game.legal_moves:
                print(f"env.step: {agent} has no valid moves!")
                is_valid = self.game.process_move_ai(self._unflatten_action(action))
                return
            if agent == '1':
                action_full = self._translate_p1_action(unflat_action)
                action = self._flatten_action(*action_full)
            
            # Ready game for AI input
            print(f"action hash: {action} action tuple: {self._unflatten_action(action)}")

            readable_action = self._unflatten_action(action)
            print(f'Agent {agent} is attempting the following move: {readable_action} ({action})')
            # Ensure/assert action is legal, if so change board accordingly
            is_valid = self.game.process_move_ai(self._unflatten_action(action))

        self.infos[agent] = self._get_infos()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.win_status in [0, 1]:
            self.terminations[agent] = True

        # TODO: rendering 
        if self.render_mode == "human":
            self.render()
    
    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
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
        return self._observation_space[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def _generate_action_mask(self):
        pass

    def _flatten_action(self, origin, destination):
        return 26 * origin + destination

    def _unflatten_action(self, hash):
        origin = hash // 26
        destination = hash % 26
        return (origin, destination)

    def _get_infos(self):
        return {}

    def observe(self, agent):
        print(f"Observing {agent}")
        self.game.ready_board_ai(int(agent))
        observation_dict = {key: 0 for key in self._observation_spaces[agent]['observation'].keys()}
        current_agent = self.possible_agents.index(agent)
        board = self.game.board

        # Add spots on board
        for spot_i, spot in enumerate(board):
            if spot.player:
                observation_dict[f'spot_{spot_i}'] = spot.piece_count if str(spot.player.id) == self.agent_selection else spot.piece_count * -1
                # observation_v[spot_i] = spot.piece_count if str(spot.player.id) == self.agent_selection else spot.piece_count * -1
                # observation_v[spot_i] = spot.piece_count if str(spot.player.id) == self.agent_selection else spot.piece_count
        observation_dict['bar_p1'] = board[0]
        observation_dict['bar_p2'] = board[25]
        # Add dice
        # print(f"current_player: {self.game.current_player} agent: {self.agent_selection} dice: {self.game.dice}")
        for die_i, die in enumerate(self.game.dice):
            if die:
                # observation_v[25 + die_i] = self.game.dice[die_i]
                observation_dict[f'dice_{die_i + 1}'] = self.game.dice[die_i]
        # observation = {key: value for key, value in zip(self._observation_spaces[agent]['observation'].keys(), observation_v)}
        
        # If agent 1, flip board values
        if current_agent == 1:
            observation = self._translate_p1_observation(observation_dict)    

        # Add action mask
        action_mask = np.zeros(26*26, "int8")
        if agent == self.agent_selection:
            if agent == '0':
                legal_moves_full = [
                    (origin, destination) for origin, destination in self.game.legal_moves if self.game.dice
                    ]
            else:
                legal_moves_full = [
                    (25 - origin, 25 - destination) for origin, destination in self.game.legal_moves if self.game.dice
                    ]
            
        # Flatten legal moves for model
        legal_moves_flat = [
            self._flatten_action(origin, destination) for origin, destination in legal_moves_full
            ]
        
        for move in legal_moves_flat:
            action_mask[move] = 1
        
        return {'observation': observation_dict, 'action_mask': action_mask}
    
    """
    translate p1 agent observation for consistent policy learning. 
    Flips board values.

    accepts:
        Dict: the observation (action_mask excluded)

    returns:
        Dict: the observation as it would appear with board indexes flipped.
    """
    def _translate_p1_observation(self, observation):
        assert len(observation) > 2, 'This function accepts the observations only'
        # Switch bar space indexes 
        # TODO: find more elegant way of doing this
        new_observation = copy(observation)
        new_observation['bar_p1'] = observation['bar_p2']
        new_observation['bar_p2'] = observation['bar_p1']

        for i in range(1, 25):
            new_observation[f'spot_{i}'] = observation[f'spot_{25-i}']
        return new_observation

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
            print(f"is_valid: {is_valid}")
            return True
        return False
    
    def _calculate_reward(self, agent, end_status):
        reward = 0
        
        # TODO: Reward blocking strategy (+0.1 for every consecutive fortified position this turn
        # TODO: Reward offensive plays (+0.1 if you took an enemy piece this turn)
        longest_block = 0
        current_block = 0
        for position in self.game.board:
            # Discourage vulnerable pieces (-0.05 for each lonely checker)
            if position.player:
                if position.player.id == int(agent) and position.piece_count == 1:
                    reward -= 0.05
                if position.player.id == int(agent) and position.piece_count > 1 and position.index not in [0, 25]:
                    current_block += 1
                else:
                    current_block = 0
                if current_block > longest_block:
                    longest_block = current_block
        reward += longest_block * 0.05
        # Reward winning (+10 if you won this turn)
        if self.game.win_status == int(agent):
            reward += 10

        # discourage losing (-10 if you lost this turn)
        if self.game.win_status == int(agent)^1:
            reward -= 10

        return reward
    