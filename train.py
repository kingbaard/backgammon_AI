from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from backgammon_env import backgammon_env_v0
from backgammon_gym_env import backgammon_gym_env_v0

def train(train_params):

    match train_params["policy_type"]:
        case "mask":
            ppo_obj = PPO
            sb3_policy = "MlpPolicy"
            env = backgammon_gym_env_v0(None, True, True)
            eval_fn = None
        case "no_mask":
            ppo_obj = PPO
            sb3_policy = "MlpPolicy"
            env = backgammon_gym_env_v0(None, False, True)
            eval_fn = None
        case "mask_selfplay":
            ppo_obj = MaskablePPO
            sb3_policy = MaskableActorCriticPolicy
            env = backgammon_env_v0()
            eval_fn = None

    model = ppo_obj(sb3_policy, 
                env, 
                batch_size=train_params["batch_size"],
                n_steps=train_params["n_steps"],
                learning_rate=train_params["learning_rate"],
                tensorboard_log="./no_mask_ppo_tensorboard3/",
                verbose=1,
                device='cuda',
                )
    
    model.learn()

    model.save(f"final_policies/{train_params['output_name']}")