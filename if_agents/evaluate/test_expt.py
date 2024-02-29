from .evaluate import run_experiment
import sys
from ..agents.agents import DummyAgent

def main():
    agent = DummyAgent()
    run_experiment(
        agent, 
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        experiment_name='dummy_agent_expt',
        filtered_game_list=['zork1.z5'], 
        max_steps=10)

if __name__ == '__main__':
    main()