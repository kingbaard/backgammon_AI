import random
import re

from backgammon_env.envs.player import Player
from backgammon_env.envs.position import Position 
# from backgammon_env.envs.position import Position
# from backgammon_env.envs.player import Player

from copy import deepcopy

class Game():
    def __init__(self, p1_type, p2_type, print_output=False):
        self.print_output = print_output
        
        self.players = [Player(0, 0, p1_type), Player(1, 25, p2_type)]
        self.turn = 0
        self.legal_moves = {}
        self.current_player = None
        self.board = self._reset_board()
        self.dice = []
        self.win_status = None
        self.start_new_game()
        self.switch_board = True 

    def __str__(self):
            if not self.board:
                return "Board not yet initialized"

            # Visual representation of the board
            board_str = ""

            board_str += f"Current Player: {self.current_player}"
            # Divider
            board_str += "\n" + "-" * 76 + "\n"

            # Upper part of the board
            for i in range(13, 25):
                board_str += f"{str(self.board[i].index).zfill(2):^5} "
            board_str += "\n"

            # Populate player affiliation with location
            for i in range(13, 25):
                location = self.board[i]
                if location.player:
                    symbol = 'X' if location.player.id == 0 else 'O'
                else:
                    symbol = ' '
                board_str += f"{symbol:^5} "
            board_str += "\n"

            # Populate top of the board piece numbers
            for i in range(13, 25):
                location = self.board[i]
                board_str += f"{location.piece_count:^5} "

            # Divider
            board_str += "\n" + "-" * 76 + "\n"

            # Populate low board pice numbers
            for i in range(12, 0, -1):
                location = self.board[i]
                board_str += f"{location.piece_count:^5} "
            board_str += "\n"
            
            # Lower part of the board affiliation
            for i in range(12, 0, -1):
                location = self.board[i]
                if location.player:
                    symbol = 'X' if location.player.id == 0 else 'O'
                else:
                    symbol = ' '
                board_str += f"{symbol:^5} "
            board_str += "\n"
            
            for i in range(12, 0, -1):
                board_str += f"{str(self.board[i].index).zfill(2):^5} "

            # Divider
            board_str += "\n" + "-" * 76 + "\n"
            board_str += f"P1 Bar: {'X'*self.board[0].piece_count}\n"
            board_str += f"P2 Bar: {'O'*self.board[25].piece_count}\n"
            board_str += f"P1 Off: {self.players[0].goal}\n"
            board_str += f"P2 Off: {self.players[1].goal}\n"
            board_str += f"Dice: {self.dice}"

            return board_str

    def run_game(self):
        while True:
            self._handle_moves()
    

    def start_new_game(self):
        # Choose who goes first
        # self.current_player = self.players[0] if random.getrandbits(1) == 0 else self.players[1]
        self.current_player = self.players[0]
        self._roll_dice()


    def _handle_moves(self):
        if self.print_output:
            print(f"Current Player: {self.current_player}")
            print(self)

        if self.dice:
            if self.current_player.type == 'human':
                self.prompt_move_user()
            else:
                self.ready_board_ai(self.current_player.id)
        else:
            self._end_turn()

    def prompt_move_user(self):
        # self._switch_player()
        if self.print_output:
            print(self)
            print(f"Current Player: {self.current_player}")
        self.dice = self._roll_dice()
        while self.dice:
            self.legal_moves = self.get_possible_moves(self.current_player.id)
            print(f"dice: {self.dice}")
            if self.print_output:
                print(f"possible_moves: {self.legal_moves}")
            if len(self.legal_moves) == 0:
                print("No possible moves!")
                return
            input_move = input('Enter move:  ')
            move = tuple(map(int, re.findall(r'\d+', input_move)))
            self.dice.remove(abs(move[0] - move[1]))
            self._move_piece(move)

    def ready_board_ai(self, player_id):
        self.legal_moves = self.get_possible_moves(player_id) if int(player_id) == self.current_player.id else set()
        if len(self.legal_moves) == 0:
            print("No possible moves! End of Turn!")
            if player_id == self.current_player.id:
                self._end_turn()
            return

    def process_move_ai(self, move):
        if move in self.legal_moves:
            self._clear_used_die(move)
            self._move_piece(move)
            if not self.dice:
                self._end_turn()
            # self.win_check()
            return True
        else:
            self._switch_player()
            self._roll_dice()
            self.turn += 1
            self.win_check()
            return False
        
    def _end_turn(self):
        self._switch_player()
        self._roll_dice()
        self.turn += 1

    def _reset_board(self):
        board = [Position(i) for i in range(26)]
        # board[0] = Position(0, 25, self.players[1], 0)
        # board[1] = Position(1, 24, self.players[0], 2)
        # board[6] = Position(6, 19, self.players[1], 5)
        # board[8] = Position(8, 17, self.players[1], 3)
        # board[12] = Position(12, 13, self.players[0], 5)
        # board[13] = Position(13, 12, self.players[1], 5)
        # board[17] = Position(17, 8, self.players[0], 3)
        # board[19] = Position(19, 6, self.players[0], 5)
        # board[24] = Position(24, 1, self.players[1], 2)
        # board[25] = Position(25, 0, self.players[0], 0)
        board[0] = Position(0, self.players[1], 0)
        board[1] = Position(1, self.players[0], 2)
        board[6] = Position(6, self.players[1], 5)
        board[8] = Position(8, self.players[1], 3)
        board[12] = Position(12, self.players[0], 5)
        board[13] = Position(13, self.players[1], 5)
        board[17] = Position(17, self.players[0], 3)
        board[19] = Position(19, self.players[0], 5)
        board[24] = Position(24, self.players[1], 2)
        board[25] = Position(25, self.players[0], 0)
        return board

    def get_possible_moves(self, player_id):
        self._can_bear_check(player_id)
        legal_moves = set() # set of (origin, destination) tuples 
        origins = set() # set of board position index ints
        player = self.players[player_id]
        # Find origins
        if self.board[self.current_player.bar_i].piece_count > 0:
            origins.add(player.bar_i) # If player has any checkers on their bar, that is the only origin
        else:
            origins.update([pos.index for pos in self.board if pos.player == player and pos.piece_count >= 0])
            # Bars cannot be origins if player has no pieces there
            if 0 in origins:
                origins.remove(0)
            if 25 in origins:
                origins.remove(25)
            print(f"observed origins: {origins}")

        # Find possible destinations
        dir = 1 if self.current_player.id == 0 else -1 
        for origin in origins:
            if self.dice:
                for die_value in self.possible_distances(self.dice):
                    destination = self._get_destination(origin + die_value * dir)
                    if destination is not None:
                        legal_moves.add((origin, destination))
                # legal_moves.update([(origin, origin + (dist * dir)) for dist in self.possible_distances(self.dice) if self._get_destination(origin + (dist * dir))])

        return legal_moves

    def _move_piece(self, move):
        origin = self.board[move[0]]
        #Decrement origin
        origin.piece_count -= 1
        if origin.piece_count == 0 and origin.index not in [0, 25]:
            origin.player = None

        # Check if bearing tile
        # if move[1] == 'b':
        if move[1] == self.players[self.current_player.id^1].bar_i:
            assert self.current_player.can_bear, f"Player {self.current_player} tried to illegally bear off!"
            self.current_player.goal += 1
            return

        # Check if eliminating opponent 
        destination = self.board[move[1]]
        if destination.player:
            if destination.player.id == self.current_player.id^1:
                assert destination.piece_count == 1
                self.board[destination.player.bar_i].piece_count += 1
                destination.player = self.current_player
                return
            
        destination.piece_count += 1
        if not destination.player:
            destination.player = self.current_player

    """
    Determines the status of potential move destinations. 

    Args:
        destination_i: the position index of the location in question. 

    Returns:
        int or char or None: destination_index as int if valid, char 'b' if move will result in bearing off, None if invalid
    """
    def _get_destination(self, destination_i):
        #If player can bear off
        if self.current_player.can_bear:
            if self.current_player.id == 0 and destination_i > 24:
                return self.players[1].bar_i
            elif self.current_player.id == 1 and destination_i < 1:
                return self.players[0].bar_i
        
        # Invalid if out of bounds
        if not 0 < destination_i < 24:
            return None
        
        elif self.board[destination_i].player in [None, self.current_player]:
            return destination_i
        if self.board[destination_i].player != self.current_player and \
            self.board[destination_i].piece_count == 1:
                return destination_i
        return None

    def _can_bear(self, player_id):
        pieces = [p for p in self.board if p.player.id == player_id]
        for p in pieces:
            if abs(player_id.bear_loc - p.index) > 6:
                return False
        return True
    
    def _clear_used_die(self, move):
        if not self.dice:
            return
        if move[1] in [0, 25]:
            for die in self.dice:
                if die > abs(move[0] - move[1]):
                    self.dice.remove(die)
                    break
        else:
            self.dice.remove(abs(move[0] - move[1]))
    
    def _switch_player(self):
        new_id = self.current_player.id^1
        self.current_player = self.players[new_id]

    def _roll_dice(self):
        dice = [random.randint(1,6) for _ in range(2)]
        # If doubles were rolled, add two copies of that number
        if dice[0] == dice[1]:
            dice.append(dice[0])
            dice.append(dice[0])
        self.dice = dice
        return dice
    
    def _can_bear_check(self, player_id):
        if int(player_id) == 0:
            for location_i in range(1, 19):
                if self.board[location_i].player:
                    if self.board[location_i].player.id == 0:
                        self.players[0].can_bear = False
                        return False
            self.players[player_id].can_bear = True
            return True
        else:
            for location_i in range(7, 25):
                if self.board[location_i].player:
                    if self.board[location_i].player.id == 1:
                        self.players[1].can_bear = False
                        return False
            self.players[player_id].can_bear = True
            return True
    
    def flip_board(self):
        new_board = deepcopy(self.board[:])
        # Switch bar space indexes 
        # TODO: find more elegant way of doing this
        temp = self.current_player.bar_i
        self.current_player.bar_i = self.players[self.current_player.id^1].bar_i
        self.players[self.current_player.id^1].bar_i = temp 

        # Swap all pos props with counterpart
        for pos in new_board:
            pos.player = self.board[25 - pos.index].player
            pos.piece_count = self.board[25 - pos.index].piece_count
            pos.index = 25 - pos.index
        self.board = new_board
        return new_board
    
    def win_check(self):
        if self.players[self.current_player.id].goal >= 15:
            self.win_status = self.current_player.id
            return self.win_status

    @staticmethod
    def possible_distances(dice):
        if dice:
            dists = set({dice[0]})
            # Only one dice to use
            if len(dice) < 2:
                return dists
            # Both dice are available to use
            if len(dice) == 2:
                dists.add(dice[1])
                # dists.add(dice[0] + dice[1])
            # Doubles
            # else:
            #     for v in range(2, len(dice) + 1):
            #         dists.add(dice[0] * v)
            return dists
        return None
    
#test
# game = Game("human", "human", True)
# game.start_new_game()
# game.run_game()