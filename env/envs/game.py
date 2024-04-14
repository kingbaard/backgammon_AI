import random
from position import Position
from player import Player

class Game():

    def __init__(self):
        self.players = [Player(0), Player(1)] 
        self.current_player = 0
        self.board = self._reset_board()
        self._start_game()

    def __str__(self):
            if not self.board:
                return "Board not yet initilized"

            # Visual representation of the board
            board_str = ""

            # Divider
            board_str += "\n" + "-" * 76 + "\n"

            # Upper part of the board
            for i in range(12, 24):
                board_str += f"{str(i).zfill(2):^5} "
            board_str += "\n"

            # Populate player affilication with location
            for i in range(12, 24):
                location = self.board[i]
                if location.player:
                    symbol = 'X' if location.player.id == 0 else 'O'
                else:
                    symbol = ' '
                board_str += f"{symbol:^5} "
            board_str += "\n"

            # Populate top of the board piece numbers
            for i in range(12, 24):
                location = self.board[i]
                board_str += f"{location.piece_count:^5} "

            # Divider
            board_str += "\n" + "-" * 76 + "\n"

            # Populate low board pice numbers
            for i in range(11, -1, -1):
                location = self.board[i]
                board_str += f"{location.piece_count:^5} "
            board_str += "\n"
            
            # Lower part of the board affiliation
            for i in range(11, -1, -1):
                location = self.board[i]
                if location.player:
                    symbol = 'X' if location.player.id == 0 else 'O'
                else:
                    symbol = ' '
                board_str += f"{symbol:^5} "
            board_str += "\n"
            
            for i in range(11, -1, -1):
                board_str += f"{str(i).zfill(2):^5} "

            # Divider
            board_str += "\n" + "-" * 76 + "\n"
            board_str += f"P1 Bar: {'X'*self.players[0].bar}\n"
            board_str += f"P2 Bar: {'O'*self.players[1].bar}\n"

            return board_str
    
    def _start_game(self):
        # Choose who goes first
        self.current_player = self.players[0] if random.getrandbits(1) == 0 else self.players[1]
        print(f"{str(self.current_player)} goes first!")
        print(self)

    def _reset_board(self):
        board = [Position(i) for i in range(24)]
        board[0] = Position(0, self.players[0], 2)
        board[5] = Position(5, self.players[1], 5)
        board[7] = Position(7, self.players[1], 3)
        board[11] = Position(11, self.players[0], 5)
        board[12] = Position(12, self.players[1], 5)
        board[16] = Position(16, self.players[0], 3)
        board[18] = Position(18, self.players[0], 5)
        board[23] = Position(23, self.players[1], 2)

        # TODO: Add bar/goal to board (for easier board vecorization)

        return board

    def play_turn_manual(self):
        dice = self._roll_dice()
        possible_moves = self._get_possible_moves(self.current_player, dice)

    def _get_possible_moves(self, player, dice):
        legal_moves = set() # set of (origin, destination) tuples 
        origins = set() # set of board position index ints
        
        # Find origins
        for pos in self.board:
            if pos.player == player:
                origins.add(pos)

        # Find possible destinations
        dir = 1 if player.id == 0 else -1
        for pos in origins:
            for dist in self.possible_distances(dice):
                dest = pos + (dist * dir)
                if self._get_destination(dest, player):
                    legal_moves.add((pos, dest))

        return legal_moves

    def _move_piece(self, move):
        #Decrement origin
        move[0].piece_count -= 1
        if move[0].piece_count == 0:
            move[0].player = None
        
        # Check if bearing tile
        if move[1] == 'b':
            assert self.current_player.can_bear
            self.current_player.goal += 1
            return

        # Check if eliminating opponent 
        if move[1].player is not self.current_player:
            assert move[1].pice_count == 1
            move[1].player.bar += 1
            move[1].player = self.current_player
            return

    def _get_destination(self, destination_i):
        #If player can bearoff
        if self.current_player.can_bear:
            if self.current_player.id == 0 and destination_i > 23:
                return 'b'
            elif self.current_player.id == 1 and destination_i < 0:
                return 'b'
        if 0 > destination_i > 24:
            return None
        if self.boad[destination_i].player in [None, self.current_player]:
            return self.board[destination_i]
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
        if self.current_player.id == 0:
            self.current_player == 1
        else:
            self.current_player == 0

    def _roll_dice(self):
        dice = [random.randint(1,6) for _ in range(2)]
        
        # If doubles were rolled, add two copies of that number
        if dice[0] == dice[1]:
            dice.append(dice[0])
            dice.append(dice[0])
        return dice

    @staticmethod
    def possible_distances(dice):
        dists = set(dice[0])
        # Only one dice to use
        if len(dice) < 2:
            return dists
        # Both dice are avalible to use
        if len(dice) == 2:
            dists.add(dice[1])
            dists.add(dice[0] + dice[1])
        # Doubles
        else:
            for v in range(2, len(dice) + 1):
                dists.add(dice[0] * v)
        return dists
    
# test

game = Game()
