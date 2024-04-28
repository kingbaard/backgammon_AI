# Backgammon AI

## How to Run RL Implementation

- Set your interpreter/environment to be Python version 3.11 or lower
- Run grid search by executing `python grid_search.py` in Bash or Powershell
- Run Bayesian optimization by executing `python bayesian_opt.py` in Bash or Powershell
- Train and evaluate the final model by executing `maskable_ppo.py` in Bash or Powershell

Whenever training is being performed, its progress can be seen using the Tensorboard web interface. Access it by running `tensorboard --logdir maskable_ppo_tensorboad`, and navigating to `http://localhost:6006`.