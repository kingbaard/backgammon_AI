import re

def int_if_possible(val):
    try:
        return int(val)
    except:
        return val

class Player:
    def __init__(self, id, bar_i, type):
        self.id = id
        self.bar_i = bar_i
        self.bear_loc = 25 - bar_i
        self.goal = 0
        self.progress = 0
        self.type = type

    def __str__(self):
        return f"Player {self.id + 1}"

    def get_move(self, game, possible_moves):
        if self.type == 'human':
            move = None
            while not move in possible_moves:
                input_move = input('Enter move:  ')
                move = tuple(map(int_if_possible, re.findall(r'[0-9b]+', input_move)))
                print(move)

            return (move,)
        else:
            return self.type.get_move(game, possible_moves)

