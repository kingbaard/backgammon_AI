from dataclasses import dataclass
from player import Player

@dataclass
class Position:
    id: int
    player: Player = None
    piece_count: int = 0