import datetime
import os
import sys

import dspy
from tqdm import tqdm

from ..constants import ACTION_MAX_LEN
from ..utils import write_history, write_to_file, write_to_json
from ..agents.agents import ReActAgent, ReflexionAgent
from ..agents.tools import InteractiveFictionGame


def run_experiment(
        game_dir,
        model_name,
        agent_name,
        experiment_name,  
        experiments_dir='experiments',       
        filtered_game_list=None,
        debug = False,
        max_steps=100
    ):
    """
    Run an experiment with the given name, if provided.
    Playback + history for each game will be stored in experiments/{expt_name}_{expt_timestamp}/{game_name}.json
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    assert experiment_name != ''
    experiment_folder = f'{experiments_dir}/{experiment_name}_{model_name}_{timestamp}'
    os.makedirs(experiment_folder, exist_ok=True)

    play_all_games(
        experiment_folder, 
        agent_name,
        game_dir,
        filtered_game_list=filtered_game_list,
        debug=debug,
        max_steps=max_steps
    )


def play_all_games(
        logs_dir, 
        agent_name,
        game_dir,
        filtered_game_list=None,
        debug = False,
        max_steps=100
        ):
    """
    Play all games in game_dir using given agent, for a maximum of max_steps.
    Logs the game playback (human-readable text of the gameplay) and history (all info including reward, moves, score) to logs_dir.
    If filtered_game_list is provided, only play games in that list (assumes these games are in game_dir)
    """

    write_to_file(str(dspy.settings), f'{logs_dir}/dspy_settings.txt')
    
    if filtered_game_list:
        game_list = filtered_game_list
    else:
        game_list = os.listdir(game_dir)
    for filename in tqdm(game_list):
        print(f'Playing {filename}...')
        # play game
        try:
            playback, history = play_game(filename, agent_name, logs_dir, game_dir, debug, max_steps)
        except Exception as e:
            write_to_file(f'Error playing {filename}: {e}', f'{logs_dir}/{filename}_error.txt')
            continue


def play_game(
        filename, 
        agent_name,
        logs_dir, 
        game_dir,
        debug = False,
        max_steps=20, 
        ):
    """
    Play one game (specified by filename) using given agent, for a maximum of max_steps.
    Returns a list of dictionaries, each containing:
    * observation
    * reward
    * moves
    * score
    * action
    The action in each of these dictionaries was taken **after** all of the above fields were queried.
    Thus, outcomes of the action in one entry of the list are reflected in the next entry.
    """
    # use max steps as max iters for agent

    playback = []
    history = []

    jericho = InteractiveFictionGame(filename, game_dir, playback, history)

    if agent_name.lower() == 'react':
        agent = ReActAgent(max_iters=max_steps, tools=[jericho])
    elif agent_name.lower() == 'reflexion':
        agent = ReflexionAgent(reflect_interval=5, max_iters=max_steps, tools=[jericho], debug=debug)

    end_state = agent(input="You are playing an interactive fiction game. Begin the game with the action 'InteractiveFictionGame[Start]' and restart if the game ends. If you die use your experience to make a better choice.")

    print(end_state)

    write_history(f'{logs_dir}/{filename}_lm_history.txt', n=1)
    
    playback = '\n'.join(playback)
    return playback, history