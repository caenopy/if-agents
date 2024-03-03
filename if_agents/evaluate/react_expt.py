import json
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple

import dspy

from ..utils import read_from_json
from .evaluate_new import run_experiment

from ..agents.tools import GiveTime


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

    run_experiment(
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        experiment_name=args.expt_name,
        filtered_game_list=['detective.z5'], 
        model_name=args.model,
        debug=args.debug,
        max_steps=100)


if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    
    parser.add_argument('-e', '--expt_name', help='name of the experiment', default='react_detective_debug')
    parser.add_argument('-g', '--game', help='name of the game to play', default='detective.z5')
    parser.add_argument('-d', '--debug', help='enable debug mode', action='store_true', default=True)
    parser.add_argument('-m', '--model', help='name of the model to use', default='gpt-3.5-turbo')

    main(parser.parse_args())
