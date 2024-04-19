from dataclasses import dataclass
from backgammon_env.envs.player import Player

@dataclass
class Position:
    id: int
    player: Player = None
    piece_count: int = 0