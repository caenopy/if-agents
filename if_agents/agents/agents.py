import dspy
from collections import namedtuple

from ..constants import ACTION_MAX_LEN


class DummyAgent():
    def __init__(self):
        self.action = 'go north'
    
    def __call__(self, observation):
            return self.action

# Define a simple class with only an 'action' field
BasicReturn = namedtuple('Action', ['action'])

class TextGameWithHistory(dspy.Signature):
    """Generate an action in the form of a short imperative sentence for a text-based game."""

    observation = dspy.InputField(desc="the game's text response to the last action")
    history = dspy.InputField(desc="gameplay so far")
    action = dspy.OutputField(desc="expected to be a simple imperative command usually no more than a couple of words (e.g. 'go north')")

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
    

class TextGameWithCanonicalActions(dspy.Signature):
    """Generate an action for a text-based interactive fiction game. 
    Some common commands are 'look', 'take', 'drop', 'turn on', 'push', 'pull', 'go north', etc.
    """

    observation = dspy.InputField(desc="The game's text response to the last action")
    action = dspy.OutputField(desc="A simple command, usually no more than 4 words.")
class TextGame(dspy.Signature):
    """Generate an action in the form of a short imperative sentence for a text-based game."""

    observation = dspy.InputField(desc="the game's text response to the last action")
    action = dspy.OutputField(desc="expected to be in simple imperative command usually no more than 4 words (e.g. 'go north')")

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

    
