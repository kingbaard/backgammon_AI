import random
import re 
from copy import deepcopy

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from utils import MonteCarloSearchTreeNode

class Player:
    def __init__(self, game):
        self.game = game

    def get_move(self):
        raise NotImplementedError
    
class HumanPlayer(Player):
    def __init__(self, game):
        self.game = game
        self.type = "human" 

    def get_move(self):
        return self.game.prompt_move_user()

class RandomPlayer(Player):
    def __init__(self, game):
        super().__init__(self, game)
        self.type = "random" 

    def get_move(self):
        return random.choice(list(self.game.get_possible_moves()))
    
class PpoPolicyPlayer(Player):
        def __init__(self, game, policy_name, is_mask=False):
            super().__init__(self, game)
            self.type = "ppo" 
            if is_mask:
                self.model = MaskablePPO.load(f"final_policies/{policy_name}.zip")
            else:
                self.model = PPO.load(f"final_policies/{policy_name}.zip")

        def get_move(self, observation):
            vectorized_move = int(
                    self.model.predict(
                    observation, deterministic=True
                )[0]
            )
            return self._unflatten_action(vectorized_move)
        
        def _unflatten_action(self, hash):
            hash = hash + 338
            origin = hash // 26
            destination = hash % 26
            return (origin, destination)

class ExpectiminimaxPlayer:
    def __init__(self, game, max_depth=3):
        self.game = game
        self.max_depth = max_depth
        self.type = "expectiminimax" 


    def expectiminimax(self, game, depth, is_maximizing_player):
        current_player_id = game.current_player.id
        opponent_id = 1 - current_player_id

        # Terminal node or maximum depth reached
        if depth == 0 or game.win_status is not None:
            return self.evaluate_board(game)

        # Maximizer's turn
        if is_maximizing_player:
            best_value = float('-inf')
            for move in game.get_possible_moves(current_player_id):
                # Simulate the move
                cloned_game = deepcopy(game)
                cloned_game.process_move_ai(move)
                # Calculate the value recursively
                value = self.expectiminimax(cloned_game, depth - 1, False)
                best_value = max(best_value, value)
            return best_value
        else:  # Expectation node (minimizer with dice rolls)
            possible_dice_rolls = self.get_possible_dice_rolls()
            total_value = 0
            count_rolls = 0
            for dice in possible_dice_rolls:
                cloned_game = deepcopy(game)
                cloned_game.dice = list(dice)
                cloned_game.ready_board_ai(opponent_id)
                # Average out the expected values
                value = self.expectiminimax(cloned_game, depth - 1, True)
                total_value += value
                count_rolls += 1
            return total_value / count_rolls if count_rolls else 0

    def evaluate_board(self, game):
        # Simple evaluation based on the piece count
        score = 0
        for pos in game.board:
            if pos.player is not None:
                if pos.player.id == game.current_player.id:
                    score += pos.piece_count
                else:
                    score -= pos.piece_count
        return score

    def get_possible_dice_rolls(self):
        # Generate all possible dice roll outcomes
        outcomes = [(i, j) for i in range(1, 7) for j in range(1, 7)]
        return outcomes

    def get_move(self):
        # Find the best move using Expectiminimax algorithm
        best_value = float('-inf')
        best_move = None
        for move in self.game.get_possible_moves(self.game.current_player.id):
            cloned_game = deepcopy(self.game)
            cloned_game.process_move_ai(move)
            value = self.expectiminimax(cloned_game, self.max_depth, False)
            if value > best_value:
                best_value = value
                best_move = move
        return best_move

class MCTS:
    def __init__(self, game, iterations=1000):
        self.game = deepcopy(game)
        self.iterations = iterations

    def selection(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1())
        return node

    def expansion(self, node):
        possible_moves = node.game.get_possible_moves(node.game.current_player.id)
        for move in possible_moves:
            # print(move)
            new_game = deepcopy(node.game)
            new_game.process_move_ai(move)
            new_node = MonteCarloSearchTreeNode(new_game, move, node)
            node.children.append(new_node)
        return random.choice(node.children) if node.children else node

    def simulation(self, node):
        simulated_game = deepcopy(node.game)
        while simulated_game.win_status is None:
            moves = simulated_game.get_possible_moves(simulated_game.current_player.id)
            if moves:
                move = random.choice(list(moves))
                simulated_game.process_move_ai(move)
            else:
                simulated_game.end_turn()
        return 1 if simulated_game.win_status == simulated_game.current_player.id else 0

    def backpropagation(self, node, outcome):
        while node:
            node.visits += 1
            node.wins += outcome if node.game.current_player.id == node.parent.game.current_player.id else 1 - outcome
            node = node.parent
    
    def get_move(self):
        root = MonteCarloSearchTreeNode(self.game)
        for _ in range(self.iterations):
            node = self.selection(root)
            child = self.expansion(node)
            outcome = self.simulation(child)
            self.backpropagation(child, outcome)
        return max(root.children, key=lambda c: c.visits).move