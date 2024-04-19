from gymnasium.envs.registration import register
from backgammon_env.envs.backgammon_env import BackgammonEnv

register(
     id="envs/backgammon_env",
     entry_point="backgammon_env.envs:backgammon_env",
     max_episode_steps=300,
)