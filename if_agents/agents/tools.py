import os

import dspy
import jericho
import datetime

from ..constants import ACTION_MAX_LEN
from ..utils import read_from_json, write_to_file, write_to_json

from .signatures import ReadRelevantMemorySignature, WriteRelevantMemorySignature

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

    def __init__(self, filename, game_dir, playback, history, logs_dir, debug=False):
        self.env = jericho.FrotzEnv(f'{game_dir}/{filename}')
        self.filename = filename
        self.playback = playback # human-readable text of the gameplay
        self.history = history # all info including obs, actions, reward, moves, score
        self.logs_dir = logs_dir
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

        self.history.append({
            'observation': obs,
            'reward': reward,
            'moves': info['moves'],
            'score': info['score'],
            'action': action
        })

        if self.env.victory():
            self.playback += [f'Scored {info["score"]} out of {self.env.get_max_score()}']

        write_to_file('\n'.join(self.playback), f'{self.logs_dir}/{self.filename}.txt')
        write_to_json(self.history, f'{self.logs_dir}/{self.filename}.json')

        return obs
    
class FetchRelevantMemory(dspy.Module):
    def __init__(self, memory_file):
        super().__init__()
        self.memory_file = memory_file
        self.prod = dspy.Predict(ReadRelevantMemorySignature)

    def forward(self, observation):
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w') as f:
                pass
        with open(self.memory_file, 'r') as f:
            memory = f.read()
        return self.prod(observation=observation, context=memory)
    
class WriteRelevantMemory(dspy.Module):
    def __init__(self, memory_file):
        super().__init__()
        self.memory_file = memory_file
        self.prod = dspy.Predict(WriteRelevantMemorySignature)

    def forward(self, observation):
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w') as f:
                pass
        
        with open(self.memory_file, 'r') as f:
            memory = f.read()

        new_memory = self.prod(observation=observation, memorystream=memory).new_memory

        with open(self.memory_file, 'a') as f:
            f.write(new_memory) 
