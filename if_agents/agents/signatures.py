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

class ReadRelevantMemorySignature(dspy.Signature):
    observation = dspy.InputField(desc="the game's text response to the last action")
    context = dspy.InputField(desc="your memory stream")
    memory = dspy.OutputField(desc="the most relevant memory to the current observation")

class WriteRelevantMemorySignature(dspy.Signature):
    observation = dspy.InputField(desc="the game's text response to the last action")
    memorystream = dspy.InputField(desc="your memory stream")
    new_memory = dspy.OutputField(desc="if the observation contains new information that should be remembered in addition to the memorystream, this field will contain the new memory")

class ValidateActionSignature(dspy.Signature):
    prev_action = dspy.InputField(desc="the previous action taken")
    observation = dspy.InputField(desc="the game's text response to the previous action")
    invalid_actions = dspy.InputField(desc="a list of actions that are known to be invalid")
    valid_actions = dspy.InputField(desc="a list of actions that are known to be valid")
    is_valid = dspy.OutputField(desc="True if the action is valid and was successfully interpreted by the game, False if the action is invalid and the game did not understand it")

class GenerateCandidateActionsSignature(dspy.Signature):
    # observation = dspy.InputField(desc="the game's text response to the last action")
    # thought = dspy.InputField(desc="next steps to take based on last observation")
    thought = dspy.InputField(desc="next steps to take which need to be translated into actions")
    invalid_actions = dspy.InputField(desc="a list of actions that are known to be invalid")
    valid_actions = dspy.InputField(desc="a list of actions that are known to be valid")
    candidate_actions = dspy.OutputField(desc="A bracketed list of valid candidate actions to take based on the last observation and thought, for example '[InteractiveFictionGame[go north], InteractiveFictionGame[go south]]'. Each action is a few words using only simple verbs and nouns present in the environment from previous observations formatted as 'InteractiveFictionGame[<action>]'")