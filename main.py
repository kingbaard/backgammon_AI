from game_manager import GameManager
from tune import tune

import argparse
import json

parser = argparse.ArgumentParser(description='An AI-filled backgammon playground')
parser.add_argument('-play', nargs=2, metavar=('P0', 'P1'), help='Play a game between two agents')
parser.add_argument('-tune', nargs=3, metavar=('MODEL', 'GRID', 'BOUNDS'), help='Tune hyperparameters for a model')
parser.add_argument('-train', nargs=1, metavar='MODEL', help='Train a model')
args = vars(parser.parse_args())

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
        tune(args['tune'][0], args['tune'][1], args['tune'][2])
    elif args['train']:
        train(args['train'][0])
    else:
        print("No valid action specified. Use -play, -tune, or -train.")
    
    
