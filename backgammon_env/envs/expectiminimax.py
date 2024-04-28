from copy import deepcopy
from game import Game

class ExpectiMiniMax:
    def get_move(self, game, possible_moves):
        (val, moves) = self.move_search(game, 2, root=True)
        print("final:", val, moves)
        return moves


    def roll_search(self, game: Game, depth: int):
        new_game = deepcopy(game)
        new_game._switch_player()

        if depth == 0:
            return self.eval(new_game)

        val = 0
        for die1 in range(1, 7):
            for die2 in range(die1, 7):
                if die1 == die2:
                    probability = 1
                    new_game.dice = [die1 for _ in range(4)]
                else:
                    probability = 2
                    new_game.dice = [die1, die2]

                val += probability * self.move_search(new_game, depth)[0]

        return val / 36

    def move_search(self, game: Game, depth: int, root=False):
        best_val = None
        best_moves = None

        possible_moves = game._get_possible_moves(game.current_player, game.dice)
        #print(possible_moves)
        if not possible_moves:
            # next player's turn; negate result to keep current player's point of view
            return (-self.roll_search(game, depth-1), [])

        # TODO: don't search different permutations of the same set of moves
        for move in possible_moves:
            new_game = deepcopy(game)
            new_game._move_piece(move)
            if new_game.dice:
                # continue moving pieces
                (val, moves) = self.move_search(new_game, depth)
                moves.insert(0, move)
            else:
                # next player's turn; negate result to keep current player's point of view
                val = -self.roll_search(new_game, depth-1)
                moves = [move]

            if root:
                print(val, moves)
            if best_val == None or val > best_val:
                best_val = val
                best_moves = moves

        #print(depth, ":", best_val, best_moves)
        return (best_val, best_moves)

    def eval(self, game):
        # TODO: make a better eval function
        return (game.current_player.goal - game.players[game.current_player.id^1].goal) * 5 + (game.current_player.progress - game.players[game.current_player.id^1].progress)

if __name__ == "__main__":
    game = Game('human', ExpectiMiniMax(), True)
    game.run_game()
