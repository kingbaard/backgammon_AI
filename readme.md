# Backgammon AI

This primary objective of this project is to create and evaluate computer Backgammon opponents. The current algorithms used in the project scpoe are PPO, Masked_PPO and Expectminimax.

## How to Play Backgammon Against AI

Execute the following command: `python main.py -play <opponent type>`

### Available Opponent Types

-Random: will simpily play a random legal move
-PPO_Masked: policy trained using masked PPO self-play
-PPO: policy trained against random opponent
-ExpectMiniMax: will determine next move via the excpectminimax algorithm
-Human: Play hotseat multiplayer

## How to Run RL Implementation

- Set your interpreter/environment to be Python version 3.11 or lower
- Run grid search by executing `python grid_search.py` in Bash or Powershell
- Run Bayesian optimization by executing `python bayesian_opt.py` in Bash or Powershell
- Train and evaluate the final model by executing `maskable_ppo.py` in Bash or Powershell

Whenever training is being performed, its progress can be seen using the Tensorboard web interface. Access it by running `tensorboard --logdir maskable_ppo_tensorboad`, and navigating to `http://localhost:6006`.
