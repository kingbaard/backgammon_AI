import random
import re

from backgammon_env.envs.position import Position
from backgammon_env.envs.player import Player

class Game():
    def __init__(self, p1_type, p2_type, print_output=False):
        self.print_output = print_output
        
        self.players = [Player(0, 0, p1_type), Player(1, 25, p2_type)] 
        self.legal_moves = {}
        self.current_player = None
        self.board = self._reset_board()
        self.dice = []
        self.start_new_game()

    def __str__(self):
            if not self.board:
                return "Board not yet initialized"

            # Visual representation of the board
            board_str = ""

            # Divider
            board_str += "\n" + "-" * 76 + "\n"

            # Upper part of the board
            for i in range(13, 25):
                board_str += f"{str(i).zfill(2):^5} "
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
                board_str += f"{str(i).zfill(2):^5} "

            # Divider
            board_str += "\n" + "-" * 76 + "\n"
            board_str += f"P1 Bar: {'X'*self.board[0].piece_count}\n"
            board_str += f"P2 Bar: {'O'*self.board[25].piece_count}\n"

            return board_str
    

    def run_game(self):
        while True:
            self._prompt_move()
    

    def start_new_game(self):
        # Choose who goes first
        self.current_player = self.players[0] if random.getrandbits(1) == 0 else self.players[1]


    def _prompt_move(self):
        if self.print_output:
            print(f"Current Player: {self.current_player}")
            print(self)
        if self.current_player.type == 'human':
            self.prompt_move_user()
        else:
            self.prompt_move_ai()


    def prompt_move_user(self):
        self._switch_player()
        if self.print_output:
            print(self)
            print(f"Current Player: {self.current_player}")
        self.dice = self._roll_dice()
        while self.dice:
            self.legal_moves = self._get_possible_moves(self.current_player)
            if self.print_output:
                print(f"dice: {self.dice}")
                print(f"possible_moves: {self.legal_moves}")
            if len(self.legal_moves) == 0:
                print("No possible moves!")
                return
            input_move = input('Enter move:  ')
            move = tuple(map(int, re.findall(r'\d+', input_move)))
            self.dice.remove(abs(move[0] - move[1]))
            self._move_piece(move)

    def prompt_move_ai(self):
        self._switch_player()
        self.dice = self._roll_dice()
        if self.dice:
            self.legal_moves = self._get_possible_moves(self.current_player)
            if self.print_output:
                print(f"dice: {self.dice}")
                print(f"possible_moves: {self.legal_moves}")
            if len(self.legal_moves) == 0:
                print("No possible moves! End of Turn!")
                return

    def process_move_ai(self, move):
        if move in self.legal_moves:
            self.dice.remove(abs(move[0] - move[1]))
            self._move_piece(move)
            invalid_move = False
            return True
        return False


    def _reset_board(self):
        board = [Position(i) for i in range(26)]
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


    def _get_possible_moves(self, player, dice):
        legal_moves = set() # set of (origin, destination) tuples 
        origins = set() # set of board position index ints
        
        # Find origins
        if self.board[player.bar_i].piece_count > 0:
            origins.add(player.bar_i) # If player has any checkers on their bar, that is the only origin
        else:
            origins.update([pos.id for pos in self.board if pos.player == player and pos.piece_count > 0])
            # for pos in self.board:
            #     if pos.player == player:
            #         origins.add(pos.id)

        # Find possible destinations
        dir = 1 if player.id == 0 else -1
        for origin in origins:
            legal_moves.update([(origin, origin + (dist * dir)) for dist in self.possible_distances(dice) if self._get_destination(origin + (dist * dir))])
            # for dist in self.possible_distances(dice):
            #     dest = pos + (dist * dir)
            #     if self._get_destination(dest):
            #         legal_moves.add((pos, dest))

        return legal_moves


    def _move_piece(self, move):
        origin = self.board[move[0]]
        #Decrement origin
        origin.piece_count -= 1
        if origin.piece_count == 0 and origin.id not in [0, 25]:
            origin.player = None

        # Check if bearing tile
        if move[1] == 'b':
            assert self.current_player.can_bear
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
                return 'b'
            elif self.current_player.id == 1 and destination_i < 0:
                return 'b'
        
        # Invalid if out of bounds
        if not 0 <= destination_i < 24:
            return None
        
        # 
        elif self.board[destination_i].player in [None, self.current_player]:
            return destination_i
        if self.board[destination_i].player != self.current_player:
            if self.board[destination_i].piece_count == 1:
                return destination_i
        return None

    def _can_bear(self, player):
        pieces = [p for p in self.board if p.player == player]
        for p in pieces:
            if abs(player.bear_loc - p.id) > 6:
                return False
        return True
    
    def _switch_player(self):
        new_id = self.current_player.id^1
        self.current_player = self.players[new_id]

    def _roll_dice(self):
        dice = [random.randint(1,6) for _ in range(2)]
        
        # If doubles were rolled, add two copies of that number
        if dice[0] == dice[1]:
            dice.append(dice[0])
            dice.append(dice[0])
        return dice
    
    def _win_check(self):
        if self.players[self.current_player].goal >= 15:
            print(f"")
            quit()

    @staticmethod
    def possible_distances(dice):
        dists = set({dice[0]})
        # Only one dice to use
        if len(dice) < 2:
            return dists
        # Both dice are avalible to use
        if len(dice) == 2:
            dists.add(dice[1])
            # dists.add(dice[0] + dice[1])
        # Doubles
        # else:
        #     for v in range(2, len(dice) + 1):
        #         dists.add(dice[0] * v)
        return dists