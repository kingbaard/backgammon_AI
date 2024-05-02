from game_manager import GameManager
from tune import tune
from train import train
import argparse
import json

parser = argparse.ArgumentParser(description='An AI-filled backgammon playground')
parser.add_argument('-play', nargs=2, metavar=('P0', 'P1'), help='Play a game between two agents')
parser.add_argument('-tune', help='Tune hyperparameters for a model')
parser.add_argument('-train', help='Train a model')
args = vars(parser.parse_args())

config_data = ""
with open('config.json', 'r') as config_file:
    config_data = json.load(config_file)

def play(p0, p1):
    gm = GameManager('expectiminimax', 'expectiminimax')
    gm.play()
    
if __name__ == "__main__":
    if args['play']:
        p0 = args['play'][0] if len(args['play']) > 0 else 'random'
        p1 = args['play'][1] if len(args['play']) > 1 else 'random'
        play(p0, p1)

    elif args['tune']:
        tune(config_data["policy"]["type"], 
             config_data["grid_search"], 
             config_data["bayesian_optimization"])
        
    elif args['train']:
        train(config_data["train"])

    else:
        print("No valid action specified. Use -play, -tune, or -train.")
    
    
