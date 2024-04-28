import warnings
from random import random

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from maskable_ppo import train_masked_ppo, eval_masked_ppo, ActionMaskWrapper, mask_fn

from backgammon_env import backgammon_env_v0

import itertools

def grid_search(env_fn, hyperparams, steps=10_000, num_eval_episodes=20, **env_kwargs):
    # Unpack hyperparameters
    learning_rates = hyperparams['learning_rate']
    gammas = hyperparams['gamma']
    clip_ranges = hyperparams['clip_range']
    batch_sizes = hyperparams['batch_size']
    results = []

    param_combinations = itertools.product(learning_rates, gammas, clip_ranges, batch_sizes)

    # Iterate over each combination
    for lr, gamma, clip_range, batch_size in param_combinations:
        print(f"Training with lr={lr}, gamma={gamma}, clip_range={clip_range}, batch_size={batch_size}")
        
        env = env_fn(**env_kwargs)
        env = ActionMaskWrapper(env)
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
            tensorboard_log="./grid_search_results/",
            device='cuda'
        )

        try:
            model.learn(total_timesteps=steps)
        except Exception as e:
            print(f"Error during training: {e}")
            env.close()
            continue

        winrate = evaluate_policy(model, env, n_eval_episodes=num_eval_episodes, warn=False)
        results.append({
            'learning_rate': lr,
            'gamma': gamma,
            'clip_range': clip_range,
            'batch_size': batch_size,
            'win_rate': winrate,
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

# Running the grid search
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning) 
    env_fn = backgammon_env_v0.env
    env_kwargs = {}
    results = grid_search(env_fn, hyperparams, steps=5_000, num_eval_episodes=20, **env_kwargs)

    best_result = max(results, key=lambda x: x['win_rate'])
    print("Hyperparameters with highest winrate:", best_result)