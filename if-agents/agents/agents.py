# import dspy

# class CoT(dspy.Module):
#     def __init__(self):
#         super().__init__()
#         self.prog = dspy.ChainOfThought("question -> answer")

#     def forward(self, question):
#         return self.prog(question=question)


class DummyAgent():
    def __init__(self):
        self.action = 'go north'
    
    def __call__(self, observation):
            return self.action