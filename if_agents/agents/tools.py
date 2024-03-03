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

        if action == "Start":
            obs, info = self.env.reset()
            reward = 0
        else:   
            obs, reward, done, info = self.env.step(action)
        
        obs = obs.strip()

        self.playback.append(f'> {action}')
        self.playback.append(obs)

        self.history.append({
            'observation': obs,
            'reward': reward,
            'moves': info['moves'],
            'score': info['score'],
            'action': action
        })

        if self.env.victory():
            playback += [f'Scored {info["score"]} out of {self.env.get_max_score()}']

        return obs
