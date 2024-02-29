import sys
from ..utils import write_to_json, read_from_json, write_to_file
from jericho import *
from tqdm import tqdm
import os
import datetime

def run_experiment(
        agent, 
        game_dir,
        experiments_dir='experiments',
        experiment_name='',         
        filtered_game_list=None,
        max_steps=500
    ):
    """
    Run an experiment with the given name, if provided.
    Playback + history for each game will be stored in experiments/{expt_name}_{expt_timestamp}/{game_name}.json
    """

    timestamp = datetime.datetime.now().timestamp()
    if experiment_name:
        experiment_folder = f'{experiments_dir}/{experiment_name}_{timestamp}'
    else:
        experiment_folder = f'{experiments_dir}/{timestamp}'
    os.makedirs(experiment_folder, exist_ok=True)

    play_all_games(
        agent, 
        experiment_folder, 
        game_dir,
        filtered_game_list=filtered_game_list,
        max_steps=max_steps
    )


def play_all_games(
        agent, 
        logs_dir, 
        game_dir,
        filtered_game_list=None,
        max_steps=500, 
        ):
    """
    Play all games in game_dir using given agent, for a maximum of max_steps.
    Logs the game playback (human-readable text of the gameplay) and history (all info including reward, moves, score) to logs_dir.
    If filtered_game_list is provided, only play games in that list (assumes these games are in game_dir)
    """
    if filtered_game_list:
        game_list = filtered_game_list
    else:
        game_list = os.listdir(game_dir)
    for filename in tqdm(game_list):
        print(f'Playing {filename}...')
        # play game
        playback, history = play_game(filename, agent, game_dir, max_steps)

        write_to_file(playback, f'{logs_dir}/{filename}.txt')
        write_to_json(history, f'{logs_dir}/{filename}.json')


def play_game(
        filename, 
        agent, 
        game_dir,
        max_steps=500, 
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
    # initialize game
    env = FrotzEnv(f'{game_dir}/{filename}')

    observation, info = env.reset()
    reward = 0
    done = False
    playback = [observation] # human-readable game playback
    history = []

    while not done:
        # prompt agent for action
        action = agent(observation)

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

        if info['moves'] > max_steps:
            action = 'max_steps_exceeded'
            break
    if (env.game_over()):
        action = 'game_over'
    elif (env.victory()):
        action = 'victory'
    playback.append(f'> {action}')
    history.append({
        'observation': observation,
        'reward': reward,
        'info': info,
        'action': action
    })
    playback += [f'Scored {info["score"]} out of {env.get_max_score()}']
    playback = '\n'.join(playback)
    return playback, history