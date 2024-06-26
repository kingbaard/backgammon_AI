from dataclasses import dataclass
# from player import Player
from player import Player

@dataclass
class Position:
    index: int
    player: Player = None
    piece_count: int = 0