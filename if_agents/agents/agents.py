# import dspy

class DummyAgent():
    def __init__(self):
        self.action = 'go north'
    
    def __call__(self, observation):
            return self.action
    
# class TextGame(dspy.Signature):
#     """Generate an action for a text-based game."""

#     observation = dspy.InputField(desc="the game's text response to the last action")
#     action = dspy.OutputField(desc="expected to be in simple command form (imperative sentence)")

# class ReActAgent(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.prog = dspy.ReAct(TextGame, max_iters=5, num_results=3, tools=None)

#     def forward(self, question):
#         return self.prog(question=question)


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

    
# class CoT(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.prog = dspy.ChainOfThought("observation -> action")

#     def forward(self, question):
#         return self.prog(question=question)