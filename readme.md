# Backgammon AI

This primary objective of this project is to create and evaluate computer Backgammon opponents. The current algorithms used in the project scpoe are a variety of PPO implemenations, Montecarlo Tree Search, and Expectminimax.

## Setup

- Set your interpreter or venv to be between Python version 3.9 and 3.11
- Execute the following in bash or powershell: `pip install -r requirements.txt`

## How to Play Backgammon Against AI

Execute the following command: `python main.py -p1 <opponent type>`

## How to Observe AI Play Against Eachother

Execute the following command: `python main.py -p0 <opponent type> -p1 <opponent type>`

## Available Opponent Types

- **random:** will simpily play a random legal move
- **ppo_masked:** policy trained using masked PPO self-play
- **ppo:** policy trained against random opponent
- **expectminimax:** will determine next move via the excpectminimax algorithm
- **treesearch:** will determine next move via the Monte Carlo Tree Search algorithm
- **human:** Play hotseat multiplayer

## How to Create Your own backgammon PPO Policy

1. Edit the config.json file to your liking.
2. Execute the following command to run hyperparameter tuning:  `python main.py -tune <model name>`
3. Update the config.json file accordingly
4. Execute the following command to train and save to the policy folder:  `python main.py -create <model name>`

Tensorboad logging will be written to `logs/`. Access it by running `tensorboard --logdir logs/`, and navigating to `http://localhost:6006` in a web browser.

