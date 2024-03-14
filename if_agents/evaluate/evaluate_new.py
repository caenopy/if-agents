import datetime
import os
import sys

import dspy
from tqdm import tqdm

from ..constants import ACTION_MAX_LEN
from ..utils import write_history, write_to_file, write_to_json
from ..agents.agents import ReActAgent, ReflexionAgent
from ..agents.tools import InteractiveFictionGame, FetchRelevantMemory, WriteRelevantMemory, UpdateValidActions, GenerateCandidateActions
EMPTY_LIST = "[]"
CANONICAL_ACTIONS = "[InteractiveFiction[go north], InteractiveFiction[go south], InteractiveFiction[go east], InteractiveFiction[go west], InteractiveFiction[look], InteractiveFiction[examine x], InteractiveFiction[take x], InteractiveFiction[drop x], InteractiveFiction[inventory], InteractiveFiction[restart]]"


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

    if agent_name == 'reflexion_actionspace_persistent_actioncache':
        # create invalid actions file
        with open(f'{experiment_folder}/invalid_actions.txt', 'w') as f:
            f.write(EMPTY_LIST)
        # create valid actions file with canonical actions
        with open(f'{experiment_folder}/valid_actions.txt', 'w') as f:
            f.write(CANONICAL_ACTIONS)

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
        try:
            # play game
            playback, history = play_game(filename, agent_name, logs_dir, game_dir, debug, max_steps)
        except Exception as e:
            write_to_file(str(e), f'{logs_dir}/{filename}_error.txt')

        # The following is now handled in InteractiveFictionGame in tools.py
        # write_to_file(playback, f'{logs_dir}/{filename}.txt')
        # write_to_json(history, f'{logs_dir}/{filename}.json')


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

    jericho = InteractiveFictionGame(filename, game_dir, playback, history, logs_dir=logs_dir)

    if agent_name.lower() == 'react':
        agent = ReActAgent(max_iters=max_steps, tools=[jericho])
    elif agent_name.lower() == 'reflexion':
        agent = ReflexionAgent(reflect_interval=5, max_iters=max_steps, tools=[jericho], debug=debug)
    elif agent_name.lower() == 'reflexionmemory':
        agent = ReflexionAgent(reflect_interval=5, max_iters=max_steps, tools=[jericho], read_memory_tool=FetchRelevantMemory(memory_file=f'{logs_dir}/{filename}_memory.txt'), write_memory_tool=WriteRelevantMemory(memory_file=f'{logs_dir}/{filename}_memory.txt'), debug=debug)
    elif agent_name.lower() == 'reflexion_actionspace':
        # create invalid actions file
        with open(f'{logs_dir}/{filename}_invalid_actions.txt', 'w') as f:
            f.write(EMPTY_LIST)
        # create valid actions file with canonical actions
        with open(f'{logs_dir}/{filename}_valid_actions.txt', 'w') as f:
            f.write(CANONICAL_ACTIONS)
        
        agent = ReflexionAgent(
            reflect_interval=5, 
            max_iters=max_steps, 
            tools=[jericho], 
            update_valid_actions_tool=UpdateValidActions(
                invalid_actions_file=f'{logs_dir}/{filename}_invalid_actions.txt', 
                valid_actions_file=f'{logs_dir}/{filename}_valid_actions.txt'
                ), 
            generate_candidate_actions_tool=GenerateCandidateActions(
                invalid_actions_file=f'{logs_dir}/{filename}_invalid_actions.txt', 
                valid_actions_file=f'{logs_dir}/{filename}_valid_actions.txt'),
            debug=debug)
    elif agent_name.lower() == 'reflexion_actionspace_persistent_actioncache':
        # here we have created the invalid and valid actions files, and we will use them to persist the action cache across games
        agent = ReflexionAgent(
            reflect_interval=5, 
            max_iters=max_steps, 
            tools=[jericho], 
            update_valid_actions_tool=UpdateValidActions(
                invalid_actions_file=f'{logs_dir}/invalid_actions.txt', 
                valid_actions_file=f'{logs_dir}/valid_actions.txt'
                ), 
            generate_candidate_actions_tool=GenerateCandidateActions(
                invalid_actions_file=f'{logs_dir}/invalid_actions.txt', 
                valid_actions_file=f'{logs_dir}/valid_actions.txt'),
            debug=debug)

    end_state = agent(input="You are playing an interactive fiction game. Begin the game with the action 'InteractiveFictionGame[Start]' and restart if the game ends. If you die use your experience to make a better choice.")

    print(end_state)

    # write_history(f'{logs_dir}/{filename}_lm_history.txt', n=1) # this is now handled by the InteractiveFiction tool
    
    playback = '\n'.join(playback)
    return playback, history