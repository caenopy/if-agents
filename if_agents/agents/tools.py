import os

import dspy
import jericho
import datetime

from ..constants import ACTION_MAX_LEN
from ..utils import read_from_json, write_to_file, write_to_json

from .signatures import ReadRelevantMemorySignature, WriteRelevantMemorySignature, ValidateActionSignature, GenerateCandidateActionsSignature

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
        # valid_actions = self.env.get_valid_actions()
        # formatted_valid_actions = [f'InteractiveFictionGame[{act}]' for act in valid_actions]
        # obs += " Valid actions: " + ", ".join(formatted_valid_actions).strip()

        if self.debug:
            print(f'Observation: {obs}')

        self.playback.append(f'> {action}')
        self.playback.append(obs)

        valid_moves, failed_moves = self.get_valid_moves_failed_moves(info)
        deaths = self.get_deaths(info)
        unique_states = self.get_unique_states(self.env)
        num_unique_states = len(unique_states)
        prefixes_dict = self.get_prefixes_dict(action)

        self.history.append({
            'observation': obs,
            'reward': reward,
            'moves': info['moves'],
            'valid_moves': valid_moves,
            'failed_moves': failed_moves,
            'score': info['score'],
            'action': action,
            'action_prefixes': prefixes_dict,
            'deaths': deaths,
            'unique_states': unique_states,
            'num_unique_states': num_unique_states
        })

        if self.env.victory():
            self.playback += [f'Scored {info["score"]} out of {self.env.get_max_score()}']

        write_to_file('\n'.join(self.playback), f'{self.logs_dir}/{self.filename}.txt')
        write_to_json(self.history, f'{self.logs_dir}/{self.filename}.json')

        return obs
    
    def get_valid_moves_failed_moves(self, info):
        """
        Get the total number of valid and failed moves made in the game so far. Returns a tuple of (valid_moves, failed_moves).
        """
        if len(self.history) == 0:
            if info['moves'] == 0:
                # first move failed
                return 0, 1
            else:
                # first move was valid
                return 1, 0
        else:
            new_moves = info['moves'] - self.history[-1]['moves']
            if new_moves > 0:
                # valid move
                return self.history[-1]['valid_moves'] + new_moves, self.history[-1]['failed_moves']
            # failed move
            return self.history[-1]['valid_moves'], self.history[-1]['failed_moves'] + 1
        
    def get_deaths(self, info):
        """
        Check if the agent has died in the game. Returns a list of death observations.
        """
        if len(self.history) == 0:
            return []
        if (info['moves'] < self.history[-1]['moves']) or (info['score'] < self.history[-1]['score']):
            # moves or score has decreased
            # ideally this means the agent has died
            # hopefully this is the death observation
            death_obs = self.history[-1]['observation']
            deaths = self.history[-1]['deaths'].copy()
            deaths.append(death_obs)
            return deaths

        return self.history[-1]['deaths']

    def get_unique_states(self, env):
        """
        Get the list of unique states the agent has encountered so far in the game.
        """
        this_state = env.get_world_state_hash()
        if len(self.history) == 0:
            return [this_state]
        else:
            return list(set(self.history[-1]['unique_states']).union(set([this_state])))
        
    def get_prefixes_dict(self, action):
        """
        Get the dictionary of lowercase 2-word prefixes of the action and their counts.
        """
        prefix = ' '.join(action.lower().split()[:2])
        if len(self.history) == 0:
            return {prefix : 1}
        curr_prefixes = self.history[-1]['action_prefixes'].copy()
        if prefix in curr_prefixes:
            curr_prefixes[prefix] += 1
        else:
            curr_prefixes[prefix] = 1
        return curr_prefixes

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

class UpdateValidActions(dspy.Module):
    def __init__(self, invalid_actions_file, valid_actions_file):
        super().__init__()
        self.invalid_actions_file = invalid_actions_file
        self.valid_actions_file = valid_actions_file
        self.prod = dspy.Predict(ValidateActionSignature)

    def forward(self, prev_action, observation):
        with open(self.invalid_actions_file, 'r') as f:
            invalid_actions = f.read().split('\n')
        with open(self.valid_actions_file, 'r') as f:
            valid_actions = f.read().split('\n')
        is_valid = self.prod(
            prev_action=prev_action, 
            observation=observation, 
            invalid_actions=invalid_actions,
            valid_actions=valid_actions
            ).is_valid
        
        if is_valid == "True":
            # we think action is valid, write it to valid actions
            with open(self.valid_actions_file, 'a') as f:
                f.write(f'{prev_action}\n')
        elif is_valid == "False":
            # we think action is invalid, write it to invalid actions
            with open(self.invalid_actions_file, 'a') as f:
                f.write(f'{prev_action}\n')


class GenerateCandidateActions(dspy.Module):
    def __init__(self, invalid_actions_file, valid_actions_file):
        super().__init__()
        self.invalid_actions_file = invalid_actions_file
        self.valid_actions_file = valid_actions_file
        self.prod = dspy.Predict(GenerateCandidateActionsSignature)

    def forward(self, thought):
        with open(self.invalid_actions_file, 'r') as f:
            invalid_actions = f.read().split('\n')
        with open(self.valid_actions_file, 'r') as f:
            valid_actions = f.read().split('\n')
        return self.prod(
            thought=thought, 
            invalid_actions=invalid_actions,
            valid_actions=valid_actions
            ).candidate_actions