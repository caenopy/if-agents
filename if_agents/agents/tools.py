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
            playback += [f'Scored {info["score"]} out of {self.env.get_max_score()}']

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
            if new_moves >= 0:
                # valid move
                return self.history[-1]['total_moves'] + new_moves, self.history[-1]['failed_moves']
            # failed move
            return self.history[-1]['total_moves'], self.history[-1]['failed_moves'] + 1
        
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
            deaths = self.history[-1]['deaths'].append(death_obs)
            return deaths

        return self.history[-1]['deaths']

    def get_unique_states(self, env):
        """
        Get the set of unique states the agent has encountered so far in the game.
        """
        this_state = env.get_world_state_hash()
        if len(self.history) == 0:
            return set([this_state])
        else:
            return self.history[-1]['unique_states'].union(set([this_state]))
    
    def get_prefixes_dict(self, action):
        """
        Get the dictionary of lowercase 3-word prefixes of the action and their counts.
        """
        curr_prefixes = self.history[-1]['action_prefixes']
        prefix = action.lower().split()[:3]
        if prefix in curr_prefixes:
            curr_prefixes[prefix] += 1
        else:
            curr_prefixes[prefix] = 1
        return curr_prefixes