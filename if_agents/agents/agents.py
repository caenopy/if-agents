import dspy
from collections import namedtuple

from ..constants import ACTION_MAX_LEN
from .signatures import *

class DummyAgent():
    def __init__(self):
        self.action = 'go north'
    
    def __call__(self, observation):
        return self.action

# TODO(nic): depricate this, use dspy.Prediction instead
# Define a simple class with only an 'action' field
BasicReturn = namedtuple('Action', ['action'])

class AutoregressiveAgent():
    def __init__(self, llm, prompt=None, max_tokens=250, context_length=1000):
        super().__init__()
        self.max_tokens = max_tokens
        self.llm = llm
        self.context_length = context_length
        self.history = str(prompt) if prompt else ''

    def __call__(self, observation):
        self.history += observation + '\n'
        if len(self.history) > self.context_length:
            self.history = self.history[-self.context_length:]
        action = self.llm(prompt=self.history, max_tokens=self.max_tokens)[0]
        action = action.split('\n')[0] # This is kind of hacky, just take first line of output.
        action = action[:ACTION_MAX_LEN]
        self.history += action + '\n'
        return dspy.Prediction(action=action) #BasicReturn(action=action)

class BasicSlidingWindowAgent(dspy.Module):
    def __init__(self, chain_of_thought=False, history_lookback=10):
        super().__init__()
        self.generate_action = dspy.ChainOfThought(TextGameWithHistory) if chain_of_thought else dspy.Predict(TextGameWithHistory)
        self.history = [] # list of (observation, action) pairs
        self.history_lookback = history_lookback # number of rounds of most recent history to keep

    def forward(self, observation):
        if len(self.history) > self.history_lookback:
            self.history = self.history[-self.history_lookback:]
        if len(self.history) > 0:
            history_str = '\n'.join([f'Observation: {obs}\nAction: {act}' for obs, act in self.history])
        else:
            history_str = 'Begin game:\n'

        action = self.generate_action(observation=observation, history=history_str)
        dspy.Suggest(len(action.action) <= ACTION_MAX_LEN, f'Action length exceeds max length of {ACTION_MAX_LEN}. Actions should be short commands.')
        self.history.append((observation, action.action))
        return action


class ReActAgent(dspy.Module):
    def __init__(self, max_iters=5, num_results=None, tools=None):
        super().__init__()
        self.tools = tools
        self.prog = dspy.ReAct(ReActSignature, max_iters=max_iters, num_results=num_results, tools=self.tools)

    def forward(self, input):
        return self.prog(input=input)
    

# class ReflectSignature(dspy.Signature):
#     heuristic = dspy.InputField(desc="a heuristic on the validity of the most recent action")
#     reflection = dspy.OutputField(desc="reflection on the most recent action given the heuristic")


def ValidActions(action, valid_actions):
    if action in valid_actions:
        return "The most recent action was a valid action for the current game state."
    else:
        return "The most recent action was not a valid action for the current game state."

from .reflexion import Reflexion

class ReflexionAgent(dspy.Module):
    def __init__(self, reflect_interval, max_iters=5, tools=None, debug=False):
        super().__init__()
        self.tools = tools
        self.prog = Reflexion(ReActSignature, reflect_interval=reflect_interval, max_iters=max_iters, tools=self.tools, debug=debug)

    def forward(self, input):
        return self.prog(input=input)

# class Reflect(dspy.Module):
#     def __init__(self, env):
#         super().__init__()
#         self.env = env
#         self.reflect = dspy.Predict(ReflectSignature)
#         self.heuristic = ValidActions

#     def forward(self, action):
#         valid_actions = self.env.get_valid_actions()
#         heuristic = self.heuristic(action, valid_actions)
#         return self.reflect(heuristic=heuristic)

    
    
# class ReflexionAgent(dspy.Module):
#     def __init__(self, max_iters=5, num_results=None, tools=None):
#         super().__init__()
#         self.tools = tools
#         self.react = dspy.ReAct(ReActSignature, max_iters=max_iters, num_results=num_results, tools=tools)
#         self.reflect = Reflect(tools=self.tools[0].env)

#     def forward(self, action):
#         prediction = self.react(input=action)
#         reflection = self.reflect(action=prediction.action)
#         return dspy.Prediction(reflection=reflection.reflection, score=prediction.score)
    

class CoTAgent(dspy.Module):
    def __init__(self, canonicalActions=False):
        super().__init__()
        if canonicalActions:
            self.prog = dspy.ChainOfThought(TextGameWithCanonicalActions)
        else:
            self.prog = dspy.ChainOfThought(TextGame)

    def forward(self, observation):
        return self.prog(observation=observation)


# RAG example
    
# class GenerateAnswer(dspy.Signature):
# """Answer questions with short factoid answers."""

# context = dspy.InputField(desc="may contain relevant facts")
# question = dspy.InputField()
# answer = dspy.OutputField(desc="often between 1 and 5 words")


# class RAG(dspy.Module):
#     def __init__(self, num_passages=3):
#         super().__init__()

#         self.retrieve = dspy.Retrieve(k=num_passages)
#         self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
#     def forward(self, question):
#         context = self.retrieve(question).passages
#         prediction = self.generate_answer(context=context, question=question)
#         return dspy.Prediction(context=context, answer=prediction.answer)

    
