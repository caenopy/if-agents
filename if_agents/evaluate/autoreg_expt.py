import json
import os
import sys
from argparse import ArgumentParser
from collections import namedtuple

import dspy

from ..agents.agents import (AutoregressiveAgent, BasicSlidingWindowAgent,
                             CoTAgent, DummyAgent, ReActAgent)
from ..utils import read_from_json
from .evaluate import run_experiment


def main(args):

    if args.model.startswith('gpt'):
        import openai
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        turbo = dspy.OpenAI(model=args.model, max_tokens=250)
        dspy.settings.configure(lm=turbo)
    else:
        together = dspy.Together(model=args.model, max_tokens=250)
        dspy.settings.configure(lm=together)

    # agent = AutoregressiveAgent(turbo, max_tokens=250, context_length=1000)
    agent = BasicSlidingWindowAgent(history_lookback=10)

    run_experiment(
        agent, 
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        experiment_name=args.expt_name,
        filtered_game_list=['detective.z5'], 
        debug=args.debug,
        max_steps=args.steps)


if __name__ == '__main__':
    parser = ArgumentParser(description='runs an experiment with a given agent and game.')
    
    parser.add_argument('-e', '--expt_name', help='name of the experiment', default='autoreg_dspy_detective_100')
    parser.add_argument('-g', '--game', help='name of the game to play', default='detective.z5')
    parser.add_argument('-d', '--debug', help='enable debug mode', action='store_true', default=True)
    parser.add_argument('-m', '--model', help='name of the model to use', default='gpt-3.5-turbo')
    parser.add_argument('-s', '--steps', help='max steps of game', default=100)

    main(parser.parse_args())
