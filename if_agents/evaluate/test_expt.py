from .evaluate import run_experiment
import sys
import json
from ..agents.agents import DummyAgent, ReActAgent, CoTAgent
from ..utils import read_from_json

import dspy

def main():
    
    config = read_from_json('config.json')
    TOGETHER_API_KEY = config['TOGETHER_API_KEY']

    # together = dspy.Together(model="google/gemma-2b-it", max_tokens=20) # cheapest chat model on Together
    # dspy.settings.configure(lm=together)

    # print(together("What is the meaning of life?"))



    agent = CoTAgent()

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
