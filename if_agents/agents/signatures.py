import dspy


class TextGameWithHistory(dspy.Signature):
    """Use `observation` and `history` to generate an `action` for a text-based interactive fiction game. 
    Explore the environment by moving around and collecting items and try to solve the game.
    """

    history = dspy.InputField(desc="The previous 10 rounds of gameplay.")
    observation = dspy.InputField(desc="The game's text response to the last action. Your action should be based on this.")
    action = dspy.OutputField(desc="A simple action of a few words using only simple verbs and nouns present in the environment from previous observations. Some common actions are 'look', 'take', 'drop', 'turn on', 'push', 'pull', 'go north', etc.")


class TextGameWithCanonicalActions(dspy.Signature):
    """Generate an action for a text-based interactive fiction game. 
    Some common commands are 'look', 'take', 'drop', 'turn on', 'push', 'pull', 'go north', etc.
    """

    observation = dspy.InputField(desc="The game's text response to the last action")
    action = dspy.OutputField(desc="A simple command, no more than 4 words long.")


class TextGame(dspy.Signature):
    """Generate an action in the form of a short imperative sentence for a text-based game."""

    observation = dspy.InputField(desc="the game's text response to the last action")
    action = dspy.OutputField(desc="expected to be in simple imperative command usually no more than 4 words (e.g. 'go north')")

class ReActSignature(dspy.Signature):
    input = dspy.InputField(desc="An instruction to play the game.")
    score = dspy.OutputField(desc="The final score you achieve.")

class RelevantMemorySignature(dspy.Signature):
    observation = dspy.InputField(desc="the game's text response to the last action")
    context = dspy.InputField(desc="history of the game so far")
    memory = dspy.OutputField(desc="the most relevant memory to the current observation")