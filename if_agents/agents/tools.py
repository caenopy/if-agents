import dspy
import datetime

# Useless for our purposes, but it's a good example of how to write a tool
class GiveTime:
    name = "GiveTime"
    input_variable = "empty"
    desc = "takes an empty string and returns the current local time"

    def __init__(self, k=3):
        pass
   
    def __call__(self, *args, **kwargs):
        return datetime.datetime.now().strftime("%H:%M:%S")
    
