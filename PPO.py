# from __future__ import annotations
import time
import os
import glob

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from backgammon_env import backgammon_env_v0
import pettingzoo.utils
from pettingzoo.utils.conversions import aec_to_parallel

class ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    def reset(self, seed=None, options= None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        return self.observe(self.agent_selection), {}
    
    def step(self, action):
        super().step(action)
        return super().last()
    
    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().observe(self.agent_selection)["action_mask"]

def train_backgammon_ppo(
        env_fn, steps: 10_000, seed: 0, **env_kwargs
):
    env = env_fn(**env_kwargs)
    env.possible_agents = ['0', '1']
    env = ActionMaskWrapper(env)
    env = aec_to_parallel(env)
    env.reset(seed)

    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 8, num_cpus=2, base_class="stable_baselines3")

    model = PPO(
        MlpPolicy,
        env,
        verbose = 3,
        learning_rate = 1e-3,

        batch_size = 256
    )

    model.learn(steps)
    model.save(f"output_models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("saved model")
    env.close()

def eval(env_fn, num_games = 100, render_mode = None, **env_kwargs):
    env = backgammon_env_v0.raw_env(render_mode="none")

    try:
        latest_policy = max(
            glob.glob(f"output_models/{env.metadata['name']}*.zip", key=os.path.getctime)
        )
    except ValueError:
        print("Policy not found")
        exit()

    model = PPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        print(f"Game #{i} Start")
        env.reset(seed=1)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            observation, action_mask = obs.values()
            if termination or truncation:
                if env.win_status not in [0, 1]:
                    break
                winner = env.game.win_status
                scores[winner] += env._cumulative_rewards[
                    winner
                ]  # only tracks the largest reward (winner of game)
                for a in env.possible_agents:
                    total_rewards[a] += env._cumulative_rewards[a]
                round_rewards.append(env._cumulative_rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    action = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
                else:
                    action = env.action_space(agent).sample(action_mask)
                env.step(action)
            env.close

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


if __name__ == "__main__":
    env_fn = backgammon_env_v0.env
    env_kwargs = {}

    train_backgammon_ppo(env_fn, steps=10_000, seed=0, **env_kwargs)