from grid_search import run_grid_search
from bayesian_opt import run_bayesian_optimization
from backgammon_env import backgammon_env_v0
from backgammon_gym_env import backgammon_gym_env_v0
from PPO import evaluate_against_random
from maskable_ppo_selfplay import eval_masked_ppo as eval_masked_selfplay_ppo
from maskable_ppo import evaluate_against_random as eval_masked_ppo

import warnings

def tune(model_type, hyperparam_grid, bayes_param_variation):
    warnings.filterwarnings("ignore", category=UserWarning) 

    # Get policy appropriate functions
    env_fn = None
    eval_fn = None
    match model_type:
            
        case 'mask':
            env_fn = backgammon_gym_env_v0.BackgammonGymEnv
            eval_fn = eval_masked_ppo
        case 'no_mask':
            env_fn = backgammon_gym_env_v0.BackgammonGymEnv
            eval_fn = evaluate_against_random
        case 'mask_self-play':
            env_fn = backgammon_env_v0.env
            env_fn = eval_masked_selfplay_ppo
    
    search_results = run_grid_search(model_type, hyperparam_grid)

    best_grid = max(search_results, key=lambda x: x['reward'])
    print("Grid search highest performer:", best_grid)

    # Apply variation from .configs to get bayes params
    learning_rate_low = best_grid["learning_rate"] - bayes_param_variation["learning_rate"]
    learning_rate_high = best_grid["learning_rate"] + bayes_param_variation["learning_rate"]
    gamma_low = best_grid["gamma"] - bayes_param_variation["gamma"]
    gamma_high = best_grid["gamma"] + bayes_param_variation["gamma"]
    clip_range_low = best_grid["clip_range"] - bayes_param_variation["clip_range"]
    clip_range_high = best_grid["clip_range"] + bayes_param_variation["clip_range"]
    batch_size_low = best_grid["batch_size"] - bayes_param_variation["batch_size"]
    batch_size_high = best_grid["batch_size"] + bayes_param_variation["batch_size"]

    param_bounds = {
        'learning_rate': (learning_rate_low, learning_rate_high),
        'gamma': (gamma_low, gamma_high),
        'clip_range': (clip_range_low, clip_range_high),
        'batch_size': (batch_size_low, batch_size_high),
    }

    run_bayesian_optimization(env_fn, eval_fn, param_bounds)