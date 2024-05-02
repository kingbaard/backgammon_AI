from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from random import random
from PPO import train_model, evaluate_against_random
from maskable_ppo_selfplay import train_masked_ppo, eval_masked_ppo, PettingZooMaskWrapper, mask_fn
from sb3_contrib import MaskablePPO
from backgammon_env import backgammon_env_v0
from backgammon_gym_env import backgammon_gym_env_v0
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from maskable_ppo_selfplay import train_masked_ppo, eval_masked_ppo, PettingZooMaskWrapper, mask_fn
from sb3_contrib.common.maskable.evaluation import evaluate_policy as mask_eval
from PPO import evaluate_against_random as no_mask_eval



def train_and_evaluate_no_mask(eta, gamma, clip_range, batch_size):
    # Cast batch_size to int
    batch_size = int(batch_size)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        backgammon_gym_env_v0.BackgammonGymEnv(),
        learning_rate=eta,
        gamma=gamma, 
        clip_range=clip_range,
        batch_size=batch_size,
        verbose=0,
        tensorboard_log="./maskable_ppo_tensorboard/",
        device='cuda')
    
    model.learn(total_timesteps=2_000)

    return no_mask_eval(env_fn, 500)


def train_and_evaluate_mask(eta, gamma, clip_range, batch_size):
    # Cast batch_size to int
    batch_size = int(batch_size)

    env = backgammon_env_v0.env()
    env = PettingZooMaskWrapper(env)
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
    
    model.learn(total_timesteps=2_000)

    return eval_masked_ppo(env_fn, 500)

default_pbounds = {
    'eta': (75e-5, 25e-4),
    'gamma': (0.93, 0.97),
    'clip_range': (0.05, 0.15),
    'batch_size': (50, 80)
}

def run_bayesian_optimization(env, 
                              eval_fn,
                              pbounds= {
                                'eta': (75e-5, 25e-4),
                                'gamma': (0.94, 0.97),
                                'clip_range': (0.05, 0.15),
                                'batch_size': (59, 69)
                            },):
    
    eval_fn = train_and_evaluate_mask if env.metadata["name"] == "backgammon_env" else train_and_evaluate_no_mask
    bae_optimizer = BayesianOptimization(
                    f=eval_fn,
                    pbounds=pbounds,
                    random_state=1,
                    )
    
    logger = JSONLogger(path="./baye_opt_log.json")
    bae_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    bae_optimizer.maximize(
        init_points = 5,
        n_iter = 20
    )