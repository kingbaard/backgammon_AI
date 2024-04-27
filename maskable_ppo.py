import glob
import os
import time
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from random import random

from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils

from backgammon_env import backgammon_env_v0

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
    
def mask_fn(env):
    return env.action_mask()
    
def train(env_fn, steps=10_000, seed=0, **env_kwargs):
    env = env_fn(**env_kwargs)
    env = ActionMaskWrapper(env)
    env.reset(seed=seed)
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=3e-4,
        gamma=0.90, 
        clip_range=0.2,
        batch_size=64,
        verbose=3,
        tensorboard_log="./maskable_ppo_tensorboard/",
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

def eval_action_mask(env_fn, num_games=500, render_mode = None, **env_kwargs):
    env = env_fn(**env_kwargs)

    try:
        latest_policy = max(
            glob.glob(f"output_models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found")
        exit()

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
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
            env.close

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    return winrate

def train_and_evaluate(eta, gamma, clip_range, batch_size):
    # Cast batch_size to int
    batch_size = int(batch_size)

    env = backgammon_env_v0.env()
    env = ActionMaskWrapper(env)
    env.reset(seed=random())
    env = ActionMasker(env, mask_fn)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        learning_rate=eta,
        gamma=gamma, 
        clip_range=clip_range,
        batch_size=batch_size,
        verbose=0,
        tensorboard_log="./maskable_ppo_tensorboard/",
        device='cuda')
    
    model.learn(total_timesteps=5_000)

    return eval_action_mask(env_fn, 100)

pbounds = {
    'eta': (1e-4, 1e-3),
    'gamma': (0.9, 0.99),
    'clip_range': (0.1, 0.3),
    'batch_size': (40, 80)
}

bae_optimizer = BayesianOptimization(
    f=train_and_evaluate,
    pbounds=pbounds,
    random_state=1,
)

if __name__ == '__main__':
    env_fn = backgammon_env_v0.env
    env_kwargs = {}

    # train(env_fn, steps=350_000, seed=420, **env_kwargs)
    # eval_action_mask(env_fn, num_games=50, render_mode=None, **env_kwargs)
    bae_optimizer.maximize(
        init_points=5,
        n_iter=25
    )