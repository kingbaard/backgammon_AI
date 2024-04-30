# from __future__ import annotations
import time
import os
import glob

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from pettingzoo.utils.conversions import aec_to_parallel

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from backgammon_gym_env.envs.backgammon_gym_env import BackgammonGymEnv
from random import randint

# Parallel environments
env = BackgammonGymEnv()
# vec_env = make_vec_env(env, n_envs=4)

def train_model():
    model = PPO("MlpPolicy", 
                env, 
                batch_size=128,
                tensorboard_log="./no_mask_ppo_tensorboard/",
                verbose=1,
                device='cuda',
                )
    model.learn(total_timesteps=200_000)
    model.save(f"no_mask_ouput_models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

def evaluate_against_random(num_games):
    env = BackgammonGymEnv(None, True)
    policy = max(
        glob.glob(f"no_mask_ouput_models/{env.metadata['name']}*.zip"), key=os.path.getctime
    )
    model = PPO.load(policy)
    wins = [0, 0]
    total_rewards = 0
    round_rewards = []
    termination = truncation = False
    for _ in range(num_games):
        observation, info = env.reset()
        for turn in range(1_000):
            action = int(
                model.predict(
                    observation, deterministic=True
                )[0]
            )
            env.reset(seed=1)
            if termination or truncation:
                winner = env.game.win_status
                wins[winner] += 1
                round_rewards.append(reward)
                total_rewards += reward
                break

            observation, reward, termination, truncation, info = env.step(action)
    env.close
    del model 
    winrate = 0
    if sum(wins) != 0:
        winrate = wins[0] / sum(wins)
    return winrate, total_rewards, round_rewards
   
if __name__ == "__main__":
    train_model()
    print(evaluate_against_random(500))
