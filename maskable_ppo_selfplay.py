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

from backgammon_env import backgammon_env_v0

class PettingZooMaskWrapper(pettingzoo.utils.BaseWrapper):
    def reset(self, seed=None, options= None):
        super().reset(seed, options)
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]
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
    
def mask_fn(env):
    return env.action_mask()
    
def train_masked_ppo(env_fn, steps=10_000, seed=0, **env_kwargs):
    env = env_fn(**env_kwargs)
    env = PettingZooMaskWrapper(env)
    env.reset(seed=seed)
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

def eval_masked_ppo(env_fn, num_games=500, render_mode = None, **env_kwargs):
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

if __name__ == '__main__':
    env_fn = backgammon_env_v0.env
    # env_kwargs = {}
    # train_masked_ppo(env_fn, 500_000, **env_kwargs)
    winrate = eval_masked_ppo(env_fn, 500)

    print(winrate)