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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from backgammon_gym_env.envs.backgammon_gym_env import BackgammonGymEnv
from random import randint

# Parallel environments

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = BackgammonGymEnv(render_mode=None, legal_only_opponent=False, random_board=True)
        # vec_env = make_vec_env(BackgammonGymEnv)
        # vec_env.seed(seed+rank)
        # vec_env.reset()
        return env

    return _init

def train_model(envs):
    model = PPO("MlpPolicy", 
                envs, 
                batch_size=128,
                n_steps=2048,
                learning_rate=1e-4,
                tensorboard_log="./no_mask_ppo_tensorboard3/",
                verbose=1,
                device='cuda',
                )
    
    model.learn(total_timesteps=500_000)
    model.save(f"no_mask_ouput_models/{'backgammon_gym'}_{time.strftime('%Y%m%d-%H%M%S')}")

# TODO: move to seperate utils or eval file
def evaluate_against_random(num_games):
    env = BackgammonGymEnv(render_mode=None, legal_only_opponent=False, random_board=False)
    policy = max(
        glob.glob(f"output_models/{env.metadata['name']}*.zip"), key=os.path.getctime
    )
    model = PPO.load(policy)
    wins = [0, 0, 0]
    total_rewards = 0
    round_rewards = []
    for game_i in range(num_games):
        observation, info = env.reset(seed=game_i)
        termination = truncation = False
        for turn in range(1000):
            if termination or truncation:
                winner = env.game.win_status
                if winner:
                    wins[winner] += 1
                round_rewards.append(reward)
                total_rewards += reward
                break

            observation = env.observe()
            action = int(
                model.predict(
                    observation, deterministic=True
                )[0]
            )
            if turn == 9_999:
                print(env.game)
            # print(env.game)
            observation, reward, termination, truncation, info = env.step(action)
    env.close
    del model 
    winrate = 0
    if sum(wins) != 0:
        winrate = wins[0] / sum(wins)
    return total_rewards/num_games
   
if __name__ == "__main__":
    # envs = SubprocVecEnv([make_env(i, i) for i in range(4)], "spawn")
    envs = BackgammonGymEnv(render_mode=None, legal_only_opponent=False, random_board=True)
    train_model(envs)
    print(evaluate_against_random(50))
