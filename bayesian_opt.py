from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from random import random
from PPO import train_model, evaluate_against_random
from maskable_ppo import train_masked_ppo, eval_masked_ppo, ActionMaskWrapper, mask_fn
from sb3_contrib import MaskablePPO
from backgammon_env import backgammon_env_v0
from backgammon_gym_env import backgammon_gym_env_v0
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

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

    return eval_masked_ppo(env_fn, 500)


def train_and_evaluate_mask(eta, gamma, clip_range, batch_size):
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
    
    model.learn(total_timesteps=2_000)

    return eval_masked_ppo(env_fn, 500)

pbounds = {
    'eta': (75e-5, 25e-4),
    'gamma': (0.93, 0.97),
    'clip_range': (0.05, 0.15),
    'batch_size': (50, 80)
}

bae_optimizer = BayesianOptimization(
    f=train_and_evaluate,
    pbounds=pbounds,
    random_state=1,
)

if __name__ == '__main__':
    env_fn = backgammon_env_v0.env
    env_kwargs = {}
    logger = JSONLogger(path="./baye_opt_log.json")
    bae_optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    bae_optimizer.maximize(
        init_points=5,
        n_iter=30
    )