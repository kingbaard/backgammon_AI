import glob
import os
import time
from random import random

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils

from backgammon_gym_env import backgammon_gym_env_v0

class ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    def reset(self, seed=None, options= None):
        super().reset(seed, options)

    def step(self, action):
        super().step(action)
        return super().last()
    
    def observe(self):
        """Return only raw observation, removing action mask."""
        return super().observe()

    def action_mask(self):
        """Separate function used in order to access the action mask."""
        return super().get_action_mask()
    
def mask_fn(env):
    return env.get_action_mask()
    
def train_masked_ppo(env_fn, steps=10_000, seed=0, **env_kwargs):
    env = env_fn(**env_kwargs)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=1.5e-4,
        gamma=0.99, 
        clip_range=0.2,
        batch_size=64,
        verbose=3,
        tensorboard_log="output/maskable_ppo_tensorboard/",
        device='cuda')
    
    model.set_random_seed(seed)
    try:
        model.learn(total_timesteps=steps)
    except RuntimeError as e:
        print(f"Error during model learning: {e}")
        raise

    model.save(f"output_models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("saved model")
    env.close()

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=-200, warn=False)

    model.save("ppo_mask")
    del model # remove to demonstrate saving and loading

def eval_masked_ppo_pz(env_fn, num_games=500, render_mode = None, **env_kwargs):
    env = env_fn(**env_kwargs)

    try:
        latest_policy = max(
            glob.glob(f"output_models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found")
        exit()

    model = MaskablePPO.load(latest_policy)

    wins = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            observation, action_mask = obs.values()
            if termination or truncation:
                if env.game.win_status not in [0, 1]:
                    break
                winner = env.game.win_status
                # print(env.game)
                wins[winner] += 1
                for a in env.possible_agents and env._cumulative_rewards:
                    if a in env._cumulative_rewards.keys():
                        total_rewards[a] += env._cumulative_rewards[a]
                round_rewards.append(env._cumulative_rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    action = env.action_space(agent).sample(action_mask)
                else:
                    action = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
                env.step(action)
                # print(env.game)
            env.close

    # Avoid dividing by zero
    if sum(wins.values()) == 0:
        winrate = 0
    else:
        winrate = wins[env.possible_agents[1]] / sum(wins.values())
    return winrate

# TODO: move to seperate utils or eval file
def evaluate_against_random(num_games):
    env = backgammon_gym_env_v0.BackgammonGymEnv(render_mode=None, legal_only_opponent=False, random_board=False)
    policy = max(
        glob.glob(f"output_models/{env.metadata['name']}*.zip"), key=os.path.getctime
    )
    model = MaskablePPO.load(policy)
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
                    print(env.game)
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
            print(f"player 0 action: {env._unflatten_action(action)}")
            observation, reward, termination, truncation, info = env.step(action)
    env.close
    del model 
    winrate = 0
    if sum(wins) != 0:
        winrate = wins[0] / sum(wins)
    return wins, winrate

if __name__ == '__main__':
    env_fn = backgammon_gym_env_v0.BackgammonGymEnv
    env_kwargs = {}
    # train_masked_ppo(env_fn, 500_000, **env_kwargs)
    # winrate = eval_masked_ppo(env_fn, 500)
    wins, winrate = evaluate_against_random(10)

    print(winrate)