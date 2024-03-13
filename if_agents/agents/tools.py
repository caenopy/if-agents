import dspy
import jericho
import datetime

from ..constants import ACTION_MAX_LEN
from ..utils import read_from_json, write_to_file, write_to_json

# Useless for our purposes, just an example of how to write a tool
class GiveTime:
    name = "GiveTime"
    input_variable = "empty"
    desc = "takes an empty string and returns the current local time"

    def __init__(self, k=3):
        pass
   
    def __call__(self, *args, **kwargs):
        return datetime.datetime.now().strftime("%H:%M:%S")
    
class InteractiveFictionGame:
    name = "InteractiveFictionGame"
    input_variable = "a simple action consisting of a few words like 'go north', 'check inventory' or 'take the key'"
    desc = "takes a step in a text-based interactive fiction game."

    def __init__(self, filename, game_dir, playback, history, debug=False):
        self.env = jericho.FrotzEnv(f'{game_dir}/{filename}')
        self.playback = playback # human-readable text of the gameplay
        self.history = history # all info including obs, actions, reward, moves, score
        self.debug = debug

    def __call__(self, action, *args, **kwargs):

        if self.debug:
            print(f'Action: {action}')

        if len(action) > ACTION_MAX_LEN:
            print('WARNING: action length exceeds max length of {}'.format(ACTION_MAX_LEN))
            print('Truncating action to first {} characters'.format(ACTION_MAX_LEN))
            action = action[:ACTION_MAX_LEN]

        if action == "Start":
            obs, info = self.env.reset()
            reward = 0
        else:   
            obs, reward, done, info = self.env.step(action)
        
        obs = obs.strip()
        valid_actions = self.env.get_valid_actions()
        formatted_valid_actions = [f'InteractiveFictionGame[{action}]' for action in valid_actions]
        obs += " Valid actions: " + ", ".join(formatted_valid_actions).strip()

        if self.debug:
            print(f'Observation: {obs}')

        self.playback.append(f'> {action}')
        self.playback.append(obs)

        total_moves = self.get_total_moves(info)

        self.history.append({
            'observation': obs,
            'reward': reward,
            'moves': info['moves'],
            'total_moves': total_moves,
            'score': info['score'],
            'action': action
        })

        if self.env.victory():
            playback += [f'Scored {info["score"]} out of {self.env.get_max_score()}']

        return obs
    
    def get_total_moves(self, info):
        """
        Get the total number of moves made in the game so far.
        """
        if len(self.history) == 0:
            return 0
        else:
            new_moves = info['moves'] - self.history[-1]['moves']
            if new_moves >= 0:
                return self.history[-1]['total_moves'] + new_moves
            return self.history[-1]['total_moves']
