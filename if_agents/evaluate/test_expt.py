from .evaluate import run_experiment
import sys
import json
from ..agents.agents import DummyAgent, ReActAgent
from ..utils import read_from_json

import dspy

def main():
    
    config = read_from_json('config.json')
    TOGETHER_API_KEY = config['TOGETHER_API_KEY']

    together = dspy.Together(model="google/gemma-2b-it") # cheapest chat model on Together
    dspy.settings.configure(lm=together)

    agent = DummyAgent()

    run_experiment(
        agent, 
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        experiment_name='dummy_agent_expt',
        filtered_game_list=['zork1.z5'], 
        debug=True,
        max_steps=10)

if __name__ == '__main__':
    main()
