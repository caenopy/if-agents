from evaluate import run_experiment
import sys
sys.path.append('../agents')
from agents import DummyAgent

def main():
    agent = DummyAgent()
    run_experiment(
        agent, 
        '/Users/mirandamelodies99/code/if-agents/data/z-machine-games-master/jericho-game-suite',
        experiments_dir='/Users/mirandamelodies99/code/if-agents/experiments',
        experiment_name='dummy_agent_expt',
        filtered_game_list=['zork1.z5'], 
        max_steps=10)

if __name__ == '__main__':
    main()