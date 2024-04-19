from gymnasium.envs.registration import register

register(
     id="envs/Backgammon",
     entry_point="envs:GridWorldEnv",
     max_episode_steps=300,
)