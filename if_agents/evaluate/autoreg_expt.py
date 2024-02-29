from .evaluate import run_experiment
import sys
import json
from ..agents.agents import DummyAgent, ReActAgent, CoTAgent, AutoregressiveAgent, AutoregressiveDSPyAgent
from ..utils import read_from_json

from collections import namedtuple

import dspy

def main():
    
    config = read_from_json('config.json')
    TOGETHER_API_KEY = config['TOGETHER_API_KEY']

    # together = dspy.Together(model="google/gemma-2b-it", max_tokens=20) # cheapest chat model on Together
    # dspy.settings.configure(lm=together)

    # print(together("What is the meaning of life?"))

    import openai
    openai.api_key = "SECRET" # miranda personal key don't spend all her money

    turbo = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=250)
    dspy.settings.configure(lm=turbo)

    # agent = AutoregressiveAgent(turbo, max_tokens=250, context_length=1000)
    agent = AutoregressiveDSPyAgent(history_lookback=10)


    # prompt = 
    
    # print(turbo(prompt, max_tokens=250))
                    

    run_experiment(
        agent, 
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        experiment_name='autoreg_dspy_detective_100',
        filtered_game_list=['detective.z5'], 
        debug=False,
        max_steps=100)


if __name__ == '__main__':
    main()
