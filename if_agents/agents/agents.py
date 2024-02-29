import dspy
from collections import namedtuple

from ..constants import ACTION_MAX_LEN
from .signatures import *

class DummyAgent():
    def __init__(self):
        self.action = 'go north'
    
    def __call__(self, observation):
        return self.action

# Define a simple class with only an 'action' field
BasicReturn = namedtuple('Action', ['action'])

class AutoregressiveAgent():
    def __init__(self, llm, max_tokens=250, context_length=1000):
        super().__init__()
        self.max_tokens = max_tokens
        self.llm = llm
        self.context_length = context_length
        self.history = ''     

    def __call__(self, observation):
        self.history += observation + '\n'
        if len(self.history) > self.context_length:
            self.history = self.history[-self.context_length:]
        action = self.llm(prompt=self.history, max_tokens=self.max_tokens)[0]
        action = action.split('\n')[0] # This is kind of hacky, just take first line of output.
        action = action[:ACTION_MAX_LEN]
        self.history += action + '\n'
        return BasicReturn(action=action)

class AutoregressiveDSPyAgent(dspy.Module):
    def __init__(self, history_lookback=10):
        super().__init__()
        self.prog = dspy.ChainOfThought(TextGameWithHistory)
        self.history = [] # list of (observation, action) pairs
        self.history_lookback = history_lookback # number of rounds of most recent history to keep

    def forward(self, observation):
        if len(self.history) > self.history_lookback:
            self.history = self.history[-self.history_lookback:]
        if len(self.history) > 0:
            history_str = '\n'.join([f'Observation: {obs}\nAction: {act}' for obs, act in self.history])
        else:
            history_str = 'The game has just begun.'

        action = self.prog(observation=observation, history=history_str)
        self.history.append((observation, action.action))
        return action


# TODO: set up tools here, should there be a tool that takes a step in the game? what should the action be here?
class ReActAgent(dspy.Module):
    def __init__(self, max_iters=5, num_results=3, tools=None):
        super().__init__()
        self.prog = dspy.ReAct(TextGame, max_iters=max_iters, num_results=num_results, tools=tools)

    def forward(self, observation):
        return self.prog(observation=observation)
    

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

    
