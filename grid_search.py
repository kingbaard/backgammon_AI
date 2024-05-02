import warnings
from random import random
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy as mask_eval
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from maskable_ppo_selfplay import train_masked_ppo, eval_masked_ppo, PettingZooMaskWrapper, mask_fn
from stable_baselines3 import PPO

from PPO import evaluate_against_random as no_mask_eval
from backgammon_env import backgammon_env_v0
from backgammon_gym_env import backgammon_gym_env_v0

import itertools

def grid_search(env_fn, hyperparams, steps=1_000, num_eval_episodes=5, **env_kwargs):
    # Unpack hyperparameters
    learning_rates = hyperparams['learning_rate']
    gammas = hyperparams['gamma']
    clip_ranges = hyperparams['clip_range']
    batch_sizes = hyperparams['batch_size']
    results = []

    env = env_fn(**env_kwargs)

    param_combinations = itertools.product(learning_rates, gammas, clip_ranges, batch_sizes)

    # Iterate over each combination
    for lr, gamma, clip_range, batch_size in param_combinations:
        print(f"Training with lr={lr}, gamma={gamma}, clip_range={clip_range}, batch_size={batch_size}")
        
        if env.metadata['name'] == "backgammon_gym_env":
            env = env_fn(**env_kwargs)

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr,
                gamma=gamma,
                clip_range=clip_range,
                batch_size=batch_size,
                verbose=0,
                tensorboard_log="./grid_search_results_no_mask/",
                device='cuda'
            )
        
        else: 
            env = env_fn(**env_kwargs)
            env = PettingZooMaskWrapper(env)
            env.reset()
            env = ActionMasker(env, mask_fn)

            model = MaskablePPO(
                MaskableActorCriticPolicy,
                env,
                learning_rate=lr,
                gamma=gamma,
                clip_range=clip_range,
                batch_size=batch_size,
                verbose=0,
                tensorboard_log="./grid_search_results_mask/",
                device='cuda'
            )
        
        try:
            model.learn(total_timesteps=steps)
        except Exception as e:
            print(f"Error during training: {e}")
            env.close()
            continue
            
        if env.metadata['name'] == "backgammon_gym_env":
            winrate = no_mask_eval(500)
        elif env.metadata['name'] == "backgammon_env":
            winrate = mask_eval(model, env, n_eval_episodes=num_eval_episodes, warn=False)

        results.append({
            'learning_rate': lr,
            'gamma': gamma,
            'clip_range': clip_range,
            'batch_size': batch_size,
            'reward': winrate,
        })
        
        env.close()
        del model

    return results


# Our hyperparameters grid
hyperparams = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'gamma': [0.99, 0.95, 0.90],
    'clip_range': [0.1, 0.2, 0.3],
    'batch_size': [32, 64, 128]
}

def run_grid_search(env_type, hyperparams, steps, eval_episodes):
    match env_type:
        case 'mask':
            env_fn = backgammon_env_v0.env
        case 'no_mask':
            env_fn = backgammon_gym_env_v0.BackgammonGymEnv
    return grid_search(env_fn, hyperparams, steps=5_000, num_eval_episodes=20, **{})

