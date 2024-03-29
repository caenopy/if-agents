import json
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple

import dspy

from ..utils import read_from_json, get_game_list
from .evaluate_new import run_experiment

from ..agents.tools import GiveTime

# TODO: build evaluator LM which compares trajectory to walkthrough, 
# and returns some kind of reward signal

def main(args):

    if args.model.startswith('gpt'):
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        turbo = dspy.OpenAI(model=args.model, max_tokens=100)
        dspy.settings.configure(lm=turbo)
    elif args.model.startswith('gemini'):
        google = dspy.Google(model=args.model, max_output_tokens=100)
        dspy.settings.configure(lm=google) 
    else:
        together = dspy.Together(model=args.model, max_tokens=500)
        dspy.settings.configure(lm=together) 

    if (args.persistent.lower() == 'true'):
        # persist invalid / valid action files across games
        run_experiment(
            'data/z-machine-games-master/jericho-game-suite',
            experiments_dir='experiments',
            agent_name='reflexion_actionspace_persistent_actioncache',
            experiment_name=args.expt_name,
            filtered_game_list=get_game_list('possible'), 
            model_name=args.model.replace('/', '_'),
            debug=args.debug,
            max_steps=int(args.steps))
    else:
        # invalid / valid action files are generated per game
        run_experiment(
            'data/z-machine-games-master/jericho-game-suite',
            experiments_dir='experiments',
            agent_name='reflexion_actionspace',
            experiment_name=args.expt_name,
            filtered_game_list=get_game_list('possible'), 
            model_name=args.model.replace('/', '_'),
            debug=args.debug,
            max_steps=int(args.steps))

if __name__ == '__main__':
    parser = ArgumentParser(description='Reflexion experiment on Jericho.')
    
    parser.add_argument('-e', '--expt_name', help='name of the experiment', default='reflexion_possiblegames')
    parser.add_argument('-g', '--game', help='name of the game to play', default='detective.z5')
    parser.add_argument('-d', '--debug', help='enable debug mode', action='store_true', default=True)
    parser.add_argument('-m', '--model', help='name of the model to use', default='gpt-3.5-turbo')
    parser.add_argument('-s', '--steps', help='max steps', default='50')
    parser.add_argument('-p', '--persistent', help='use persistent invalid / valid action files across games', default='')

    main(parser.parse_args())
