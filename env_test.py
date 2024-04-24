from backgammon_env import backgammon_env_v0
import numpy as np
env = backgammon_env_v0.BackgammonEnv(render_mode="none")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination:
        # print("Agent terminated!")
        exit()
    action = None
    mask = None
    # print(dict(observation))
    if observation:
        mask = observation["action_mask"]
    action = env.action_space(agent).sample(mask)
    env.step(action)
    print("\n" + str(env.game))
    print(f"rewards: {env.rewards}")
env.close()