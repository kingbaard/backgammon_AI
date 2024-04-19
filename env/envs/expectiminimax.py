from math import inf
from copy import deepcopy
from game import Game

class ExpectiMiniMax:
    def roll_search(self, game: Game, depth: int):
        if depth == 0:
            return self.eval()

        val = 0
        for die1 in range(1, 7):
            for die2 in range(die1, 7):
                if die1 == die2:
                    probability = 1
                    game.dice = [die1 for _ in range(4)]
                else:
                    probability = 2
                    game.dice = [die1, die2]

                val += probability * self.move_search(game, depth)
        
        return val / 36

    def move_search(self, game: Game, depth: int):
        best_val = None
        best_moves = None

        possible_moves = game._get_possible_moves(game.current_player, game.dice)
        if not possible_moves:
            return (-self.roll_search(game, depth-1), [])

        # TODO: don't search different permutations of the same moves
        for move in possible_moves:
            new_game = deepcopy(game)
            new_game._move_piece(move)
            if new_game.dice:
                # continue moving pieces
                (val, moves) = self.move_search(new_game, depth)
                moves.insert(0, move)
            else:
                # next player's turn; negate result to keep current player's point of view
                val = -self.roll_search(game, depth-1)
                moves = [move]
            
            if best_val == None or val > best_val:
                best_val = val
                best_moves = moves

        return (best_val, best_moves)

    def eval(self, game):
        # TODO: make a better eval function
        return game.current_player.goal - game.players[game.current_player.id^1]
