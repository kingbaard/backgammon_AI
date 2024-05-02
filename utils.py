import glob
import math
import os

class MonteCarloSearchTreeNode:
    def __init__(self, game, move=None, parent=None):
        self.game = game
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def ucb1(self, exploration_weight=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

def evaluate_against_random(num_games, env_fn, **kwargs):
    env = env_fn(**kwargs)
    policy = max(
        glob.glob(f"no_mask_ouput_models/{env.metadata['name']}*.zip"), key=os.path.getctime
    )
    model = PPO.load(policy)
    wins = [0, 0, 0]
    total_rewards = 0
    round_rewards = []
    termination = truncation = False
    for _ in range(num_games):
        observation, info = env.reset()
        for turn in range(1_000):
            observation = env.observe()
            action = int(
                model.predict(
                    observation, deterministic=True
                )[0]
            )
            if turn == 9_999:
                print(env.game)
            if termination or truncation:
                winner = env.game.win_status
                if winner:
                    wins[winner] += 1
                round_rewards.append(reward)
                total_rewards += reward
                env.reset(seed=1)
                break

            observation, reward, termination, truncation, info = env.step(action)
    env.close
    del model 
    winrate = 0
    if sum(wins) != 0:
        winrate = wins[0] / sum(wins)
    return winrate, total_rewards, round_rewards