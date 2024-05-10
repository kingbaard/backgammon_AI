#Need dictionary of policies and their file locations (e.g "easy" -> path)
from game import Game
from player_classes import ExpectiminimaxPlayer, RandomPlayer, PpoPolicyPlayer, MonteCarloSearchTreeNode, HumanPlayer

#TODO: find a better home for this
policy_map = {
    "masked_PPO_selfplay": 'masked_PPO_selfplay',
    "masked_PPO" : 'masked_PPO',
    "unmasked_PPO" : 'unmasked_PPO'
    }

class GameManager():
    def __init__(self, p0_type='human', p1_type='random', p0_options=None, p1_options=None):
        self.game = Game(p0_type, p1_type)
        self.agents = [None, None]

        self.opponent = None
        for agent_i in [0, 1]:
            match [p0_type, p1_type][agent_i]:
                case "human":
                    self.agents[agent_i] = HumanPlayer(self.game)
                case "random":
                    self.agents[agent_i] = RandomPlayer(self.game)
                case "ppo_self_play":
                    self.agents[agent_i] = PpoPolicyPlayer(self.game, policy_map['masked_PPO_selfplay'], True)
                case "ppo_masked":
                    self.agents[agent_i] = PpoPolicyPlayer(self.game, policy_map['masked_PPO_selfplay'], True)
                case "ppo":
                    self.agents[agent_i] = PpoPolicyPlayer(self.game, policy_map['masked_PPO_selfplay'], False)
                case "expectiminimax":
                    self.agents[agent_i] = ExpectiminimaxPlayer(self.game, 2)
                case "treesearch":
                    self.agent[agent_i] = MonteCarloSearchTreeNode(self.game)

    def play(self):
        while not self.game.win_status:
            while self.game.dice:
                if self.agents[self.game.current_player.id]:
                    if self.game.ready_board_ai(self.game.current_player.id):
                        move = self.agents[self.game.current_player.id].get_move()
                        print(f"P{self.game.current_player.id} is making the move {move}")
                        self.game.process_move_ai(move)
                        print(self.game)
            self.game.end_turn()
        print(f"Player {self.game.win_status} Wins!")
        exit()