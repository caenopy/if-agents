import datetime
import os
import sys

import dspy
from jericho import *
from tqdm import tqdm

from ..constants import ACTION_MAX_LEN
from ..utils import read_from_json, write_history, write_to_file, write_to_json, get_game_list

def main():
    run_walkthrough(
        'data/z-machine-games-master/jericho-game-suite',
        experiments_dir='experiments',
        filtered_game_list=get_game_list('possible'), 
    )

def run_walkthrough(
        game_dir,
        experiments_dir='experiments',
        experiment_name='walkthroughs',
        filtered_game_list=None,
    ):
    """
    Run an experiment with the given name, if provided.
    Playback + history for each game will be stored in experiments/{expt_name}_{expt_timestamp}/{game_name}.json
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        experiment_folder = f'{experiments_dir}/{experiment_name}_{timestamp}'
    else:
        experiment_folder = f'{experiments_dir}/{timestamp}'
    os.makedirs(experiment_folder, exist_ok=True)

    play_all_games(
        experiment_folder, 
        game_dir,
        filtered_game_list=filtered_game_list,
    )


def play_all_games(
        logs_dir, 
        game_dir,
        filtered_game_list=None,
        ):
    """
    Play all games in game_dir using walkthrough.
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
        playback, history = play_game(filename, game_dir)

        write_to_file(playback, f'{logs_dir}/{filename}.txt')
        write_to_json(history, f'{logs_dir}/{filename}.json')


def play_game(
        filename, 
        game_dir,
        ):
    """
    Play one game (specified by filename) using walkthrough, for a maximum of max_steps.
    Returns a list of dictionaries, each containing:
    * observation
    * reward
    * moves
    * score
    * action
    The action in each of these dictionaries was taken **after** all of the above fields were queried.
    Thus, outcomes of the action in one entry of the list are reflected in the next entry.
    """
    # initialize game
    env = FrotzEnv(f'{game_dir}/{filename}')

    observation, info = env.reset()
    walkthrough = env.get_walkthrough()

    reward = 0
    playback = [observation] # human-readable game playback
    history = [] # list of dictionaries, each containing observation, reward, moves, score, action
    step = 0
    for action in walkthrough:
        step += 1
        playback.append(f'> {action}')
        history.append({
            'observation': observation,
            'reward': reward,
            'moves': info['moves'],
            'score': info['score'],
            'action': action
        })
        # Take an action in the environment using the step fuction.
        # The resulting text-observation, reward, and game-over indicator is returned.
        observation, reward, done, info = env.step(action)
        playback.append(observation)
    if (env.game_over()):
        action = 'game_over'
    elif (env.victory()):
        action = 'victory'
    playback.append(f'> {action}')
    history.append({
        'observation': observation,
        'reward': reward,
        'moves': info['moves'],
        'score': info['score'],
        'action': action
    })
    playback += [f'Scored {info["score"]} out of {env.get_max_score()}']
    playback = '\n'.join(playback)
    return playback, history

main()